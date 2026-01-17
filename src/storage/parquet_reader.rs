// Parquet file reading

use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::{ArrowReader, ParquetFileArrowReader};
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Configuration for reading Parquet files
#[derive(Debug, Clone)]
pub struct ParquetReaderConfig {
    /// Whether to read row groups in parallel (default: true)
    pub parallel: bool,
    /// Optional list of column indices to read (for column pruning)
    /// If None, all columns are read
    pub column_indices: Option<Vec<usize>>,
    /// Batch size for reading (default: 8192)
    pub batch_size: usize,
}

impl Default for ParquetReaderConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            column_indices: None,
            batch_size: 8192,
        }
    }
}

/// Parquet reader that reads files into Arrow RecordBatches
pub struct ParquetReader {
    file_path: PathBuf,
    file_reader: Arc<SerializedFileReader<BufReader<File>>>,
    arrow_reader: ParquetFileArrowReader,
    config: ParquetReaderConfig,
}

impl ParquetReader {
    /// Create a new Parquet reader from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_path_with_config(path, ParquetReaderConfig::default())
    }

    /// Create a new Parquet reader from a file path with configuration
    pub fn from_path_with_config<P: AsRef<Path>>(
        path: P,
        config: ParquetReaderConfig,
    ) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path)?;
        let buf_reader = BufReader::new(file);
        let file_reader = Arc::new(SerializedFileReader::new(buf_reader)?);
        let arrow_reader = ParquetFileArrowReader::new(file_reader.clone());

        Ok(Self {
            file_path,
            file_reader,
            arrow_reader,
            config,
        })
    }

    /// Get the Arrow schema from the Parquet file
    pub fn schema(&self) -> Result<Schema> {
        self.arrow_reader.get_schema().map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("Failed to read Parquet schema: {}", e),
            )
        })
    }

    /// Read all data from the Parquet file into RecordBatches
    /// If parallel is enabled, reads row groups in parallel
    pub fn read_all(&mut self) -> Result<Vec<RecordBatch>> {
        let num_row_groups = self.file_reader.num_row_groups();

        if num_row_groups == 0 {
            return Ok(Vec::new());
        }

        if self.config.parallel && num_row_groups > 1 {
            self.read_all_parallel()
        } else {
            self.read_all_sequential()
        }
    }

    /// Read all row groups sequentially
    fn read_all_sequential(&mut self) -> Result<Vec<RecordBatch>> {
        let mut all_batches = Vec::new();
        let num_row_groups = self.file_reader.num_row_groups();

        for i in 0..num_row_groups {
            let batches = self.read_row_group(i)?;
            all_batches.extend(batches);
        }

        Ok(all_batches)
    }

    /// Read all row groups in parallel using Rayon
    fn read_all_parallel(&mut self) -> Result<Vec<RecordBatch>> {
        let num_row_groups = self.file_reader.num_row_groups();
        let file_path = self.file_path.clone();
        let config = self.config.clone();

        // Read each row group in parallel
        let batch_results: Vec<Result<Vec<RecordBatch>>> = (0..num_row_groups)
            .into_par_iter()
            .map(|i| {
                // Each parallel task needs its own file reader
                let file = File::open(&file_path)?;
                let buf_reader = BufReader::new(file);
                let file_reader = SerializedFileReader::new(buf_reader)?;
                let arrow_reader = ParquetFileArrowReader::new(Arc::new(file_reader));

                read_row_group_parallel(arrow_reader, i, &config)
            })
            .collect();

        // Flatten results and handle errors
        let mut all_batches = Vec::new();
        for result in batch_results {
            let batches = result?;
            all_batches.extend(batches);
        }

        Ok(all_batches)
    }

    /// Read a specific row group
    pub fn read_row_group(&mut self, row_group_index: usize) -> Result<Vec<RecordBatch>> {
        let mut record_batch_reader = self
            .arrow_reader
            .get_record_reader(self.config.batch_size)
            .map_err(|e| {
                Error::new(
                    ErrorKind::Other,
                    format!("Failed to create record batch reader: {}", e),
                )
            })?;

        // Select specific row group
        record_batch_reader
            .with_row_group(row_group_index)
            .map_err(|e| {
                Error::new(
                    ErrorKind::Other,
                    format!("Failed to select row group {}: {}", row_group_index, e),
                )
            })?;

        // Apply column pruning if specified
        if let Some(ref column_indices) = self.config.column_indices {
            record_batch_reader
                .with_projection(column_indices.clone())
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Other,
                        format!("Failed to apply column projection: {}", e),
                    )
                })?;
        }

        let mut batches = Vec::new();
        loop {
            match record_batch_reader.next_batch() {
                Ok(Some(batch)) => {
                    // Validate and convert to our supported types
                    let validated_batch = validate_record_batch(batch)?;
                    batches.push(validated_batch);
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(Error::new(
                        ErrorKind::Other,
                        format!("Error reading batch: {}", e),
                    ))
                }
            }
        }

        Ok(batches)
    }

}

/// Helper function to read a row group in parallel (used by parallel reading)
fn read_row_group_parallel(
    arrow_reader: ParquetFileArrowReader<SerializedFileReader<BufReader<File>>>,
    row_group_index: usize,
    config: &ParquetReaderConfig,
) -> Result<Vec<RecordBatch>> {
    let mut record_batch_reader = arrow_reader
        .get_record_reader(config.batch_size)
        .map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("Failed to create record batch reader: {}", e),
            )
        })?;

    record_batch_reader
        .with_row_group(row_group_index)
        .map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("Failed to select row group {}: {}", row_group_index, e),
            )
        })?;

    if let Some(ref column_indices) = config.column_indices {
        record_batch_reader
            .with_projection(column_indices.clone())
            .map_err(|e| {
                Error::new(
                    ErrorKind::Other,
                    format!("Failed to apply column projection: {}", e),
                )
            })?;
    }

    let mut batches = Vec::new();
    loop {
        match record_batch_reader.next_batch() {
            Ok(Some(batch)) => {
                let validated_batch = validate_record_batch(batch)?;
                batches.push(validated_batch);
            }
            Ok(None) => break,
            Err(e) => {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Error reading batch: {}", e),
                ))
            }
        }
    }

    Ok(batches)
}

/// Validate that a RecordBatch contains only supported data types
/// and convert if necessary (returns the batch as-is if already valid)
fn validate_record_batch(batch: RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();

    // Check if all columns use supported types
    for field in schema.fields() {
        if !is_supported_type(field.data_type()) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "Unsupported data type: {:?} in column '{}'",
                    field.data_type(),
                    field.name()
                ),
            ));
        }
    }

    // If all types are supported, return as-is
    Ok(batch)
}

/// Check if a data type is supported
fn is_supported_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int32
            | DataType::Int64
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Boolean
    )
}

/// Convenience function to read a Parquet file into RecordBatches
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<RecordBatch>> {
    let mut reader = ParquetReader::from_path(path)?;
    reader.read_all()
}

/// Convenience function to read a Parquet file with configuration
pub fn read_parquet_with_config<P: AsRef<Path>>(
    path: P,
    config: ParquetReaderConfig,
) -> Result<Vec<RecordBatch>> {
    let mut reader = ParquetReader::from_path_with_config(path, config)?;
    reader.read_all()
}

// Tests can be added later with tempfile in dev-dependencies
// Example test structure:
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tempfile::TempDir;
//     // ... test implementations ...
// }
