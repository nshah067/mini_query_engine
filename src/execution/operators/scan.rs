// Scan Parquet files

use crate::execution::batch::{RecordBatch, SchemaRef};
use crate::execution::operators::Operator;
use crate::storage::parquet_reader::{ParquetReader, ParquetReaderConfig};
use arrow::datatypes::Schema;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Scan operator that reads data from Parquet files
/// Supports column projection and can read row groups in parallel
pub struct ScanOperator {
    path: PathBuf,
    projection: Option<Vec<String>>,
    schema: SchemaRef,
    config: ParquetReaderConfig,
}

impl ScanOperator {
    /// Create a new Scan operator
    /// 
    /// # Arguments
    /// * `path` - Path to the Parquet file to scan
    /// * `projection` - Optional list of column names to read (for column pruning)
    /// 
    /// # Returns
    /// Result containing the ScanOperator, or an error string
    pub fn new<P: AsRef<Path>>(path: P, projection: Option<Vec<String>>) -> Result<Self, String> {
        // Read schema first to validate the file
        let mut reader = ParquetReader::from_path(&path)
            .map_err(|e| format!("Failed to open Parquet file: {}", e))?;
        
        let arrow_schema = reader.schema()
            .map_err(|e| format!("Failed to read Parquet schema: {}", e))?;
        
        // If projection is specified, create a projected schema (prune the columns)
        let schema = if let Some(ref columns) = projection {
            let fields: Vec<_> = columns
                .iter()
                .map(|name| {
                    arrow_schema
                        .fields()
                        .iter()
                        .find(|f| f.name() == name)
                        .ok_or_else(|| format!("Column '{}' not found in schema", name))
                        .map(|f| f.clone())
                })
                .collect::<Result<_, _>>()?;
            Arc::new(Schema::new(fields))
        } else {
            Arc::new(arrow_schema)
        };

        // Determine column indices for projection if needed
        let column_indices = projection.as_ref().map(|columns| {
            columns
                .iter()
                .filter_map(|name| {
                    arrow_schema
                        .fields()
                        .iter()
                        .position(|f| f.name() == name)
                })
                .collect()
        });

        let config = ParquetReaderConfig {
            parallel: true,
            column_indices,
            batch_size: 8192,
        };

        Ok(Self {
            path: path.as_ref().to_path_buf(),
            projection,
            schema,
            config,
        })
    }

    /// Read all data from the Parquet file
    /// This is the main execution method for Scan
    pub fn read_all(&self) -> Result<Vec<RecordBatch>, String> {
        let mut reader = ParquetReader::from_path_with_config(&self.path, self.config.clone())
            .map_err(|e| format!("Failed to create Parquet reader: {}", e))?;
        
        let arrow_batches = reader.read_all()
            .map_err(|e| format!("Failed to read Parquet data: {}", e))?;

        // Convert Arrow RecordBatches to our RecordBatch type
        let batches: Vec<RecordBatch> = arrow_batches
            .into_iter()
            .map(RecordBatch::from_arrow)
            .collect();

        Ok(batches)
    }
}

impl Operator for ScanOperator {
    /// Execute the scan operator
    /// Note: Scan is a source operator, so it doesn't take input batches
    /// Instead, it reads from the file system
    /// 
    /// For compatibility with the Operator trait, we ignore the input
    /// and read from the file. In practice, Scan should be handled specially
    /// by the executor since it's a source operator.
    fn execute(&self, _input: &RecordBatch) -> Result<RecordBatch, String> {
        // Scan is a source operator - it doesn't process input batches
        // This method is called for compatibility, but Scan should be handled
        // specially by the executor
        Err("Scan operator cannot execute on input batches. Use read_all() instead.".to_string())
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
