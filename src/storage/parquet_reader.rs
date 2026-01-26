// Parquet file reading

use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch as ArrowRecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

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
/// Uses parquet 50 API with ParquetRecordBatchReaderBuilder
pub struct ParquetReader {
    file_path: PathBuf,
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
        Ok(Self { file_path, config })
    }

    /// Get the Arrow schema from the Parquet file
    pub fn schema(&self) -> Result<Schema> {
        let file = File::open(&self.file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Parquet: {}", e)))?;
        Ok(builder.schema().as_ref().clone())
    }

    /// Read all data from the Parquet file into RecordBatches
    /// If parallel is enabled, reads row groups in parallel
    pub fn read_all(&self) -> Result<Vec<ArrowRecordBatch>> {
        let file = File::open(&self.file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Parquet: {}", e)))?;

        let num_row_groups = builder.metadata().num_row_groups();

        if num_row_groups == 0 {
            return Ok(Vec::new());
        }

        if self.config.parallel && num_row_groups > 1 {
            self.read_all_parallel(num_row_groups)
        } else {
            self.read_all_sequential(builder)
        }
    }

    /// Read all row groups sequentially
    fn read_all_sequential(
        &self,
        builder: ParquetRecordBatchReaderBuilder<File>,
    ) -> Result<Vec<ArrowRecordBatch>> {
        let builder = if let Some(ref indices) = self.config.column_indices {
            let mask = ProjectionMask::leaves(builder.parquet_schema(), indices.clone());
            builder.with_projection(mask)
        } else {
            builder
        };
        let reader = builder
            .with_batch_size(self.config.batch_size)
            .build()
            .map_err(|e| Error::new(ErrorKind::Other, format!("Parquet build: {}", e)))?;

        let batches: Vec<ArrowRecordBatch> = reader
            .map(|b| b.map_err(|e| Error::new(ErrorKind::Other, format!("Parquet read: {}", e))))
            .collect::<Result<Vec<_>>>()?;

        let mut out = Vec::new();
        for batch in batches {
            out.push(validate_record_batch(batch)?);
        }
        Ok(out)
    }

    /// Read all row groups in parallel using Rayon
    fn read_all_parallel(&self, num_row_groups: usize) -> Result<Vec<ArrowRecordBatch>> {
        let file_path = self.file_path.clone();
        let column_indices = self.config.column_indices.clone();
        let batch_size = self.config.batch_size;

        let batch_results: Vec<Result<Vec<ArrowRecordBatch>>> = (0..num_row_groups)
            .into_par_iter()
            .map(|i| {
                let file = File::open(&file_path)?;
                let b = ParquetRecordBatchReaderBuilder::try_new(file)
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Parquet: {}", e)))?;
                let b = if let Some(ref ind) = column_indices {
                    let mask = ProjectionMask::leaves(b.parquet_schema(), ind.clone());
                    b.with_projection(mask)
                } else {
                    b
                };
                let r = b
                    .with_row_groups(vec![i])
                    .with_batch_size(batch_size)
                    .build()
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Parquet build: {}", e)))?;
                let batches: Vec<ArrowRecordBatch> = r
                    .map(|b| {
                        b.map_err(|e| {
                            Error::new(ErrorKind::Other, format!("Parquet read: {}", e))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let validated: Result<Vec<_>> = batches
                    .into_iter()
                    .map(validate_record_batch)
                    .collect();
                validated
            })
            .collect();

        let mut all_batches = Vec::new();
        for result in batch_results {
            let batches = result?;
            all_batches.extend(batches);
        }
        Ok(all_batches)
    }
}

/// Validate that a RecordBatch contains only supported data types
fn validate_record_batch(batch: ArrowRecordBatch) -> Result<ArrowRecordBatch> {
    let schema = batch.schema();
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
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<ArrowRecordBatch>> {
    let reader = ParquetReader::from_path(path)?;
    reader.read_all()
}

/// Convenience function to read a Parquet file with configuration
pub fn read_parquet_with_config<P: AsRef<Path>>(
    path: P,
    config: ParquetReaderConfig,
) -> Result<Vec<ArrowRecordBatch>> {
    let reader = ParquetReader::from_path_with_config(path, config)?;
    reader.read_all()
}
