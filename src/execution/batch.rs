// Batch/vector data structure

use arrow::array::ArrayRef;
use arrow::record_batch::RecordBatch as ArrowRecordBatch;
use std::sync::Arc;
pub use arrow::datatypes::{Schema, SchemaRef};

/// RecordBatch wraps Arrow's columnar data format for vectorized execution
/// Provides an abstraction layer over Arrow's RecordBatch for later extensions
#[derive(Clone, Debug)]
pub struct RecordBatch {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    num_rows: usize,
}

impl RecordBatch {
    /// Create a new RecordBatch from a schema and columns
    /// 
    /// # Arguments
    /// * `schema` - The schema describing the columns
    /// * `columns` - Vector of Arrow arrays, one per column
    /// 
    /// # Errors
    /// Returns an error if the number of columns doesn't match the schema,
    /// or if column lengths are inconsistent
    pub fn try_new(
        schema: SchemaRef,
        columns: Vec<ArrayRef>,
    ) -> Result<Self, String> {
        if schema.fields().len() != columns.len() {
            return Err(format!(
                "Schema has {} fields but {} columns provided",
                schema.fields().len(),
                columns.len()
            ));
        }

        // Check that all columns have the same length
        let num_rows = columns.first().map(|col| col.len()).unwrap_or(0);
        for (idx, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(format!(
                    "Column {} has length {} but expected {}",
                    idx,
                    col.len(),
                    num_rows
                ));
            }
        }

        Ok(Self {
            schema,
            columns,
            num_rows,
        })
    }

    /// Create a new RecordBatch from an ArrowRecordBatch
    pub fn from_arrow(batch: ArrowRecordBatch) -> Self {
        Self {
            schema: batch.schema(),
            columns: batch.columns().to_vec(),
            num_rows: batch.num_rows(),
        }
    }

    /// Convert this RecordBatch to an Arrow RecordBatch
    pub fn to_arrow(&self) -> Result<ArrowRecordBatch, String> {
        ArrowRecordBatch::try_new(self.schema.clone(), self.columns.clone())
            .map_err(|e| format!("Failed to create Arrow RecordBatch: {}", e))
    }

    /// Get the schema of this RecordBatch
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the number of rows in this batch
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns in this batch
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get a reference to the columns
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    /// Get a specific column by index
    pub fn column(&self, index: usize) -> Result<&ArrayRef, String> {
        self.columns.get(index).ok_or_else(|| {
            format!(
                "Column index {} out of bounds (batch has {} columns)",
                index,
                self.columns.len()
            )
        })
    }

    /// Get a column by name
    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        let index = self.schema.fields().iter().position(|f| f.name() == name)?;
        self.columns.get(index)
    }

    /// Select a subset of columns by indices
    /// 
    /// # Arguments
    /// * `indices` - Vector of column indices to select
    /// 
    /// # Returns
    /// A new RecordBatch containing only the selected columns
    pub fn select_columns(&self, indices: &[usize]) -> Result<Self, String> {
        let fields: Vec<_> = indices
            .iter()
            .map(|&idx| {
                self.schema
                    .fields()
                    .get(idx)
                    .ok_or_else(|| format!("Column index {} out of bounds", idx))
                    .map(|f| f.clone())
            })
            .collect::<Result<_, _>>()?;

        let columns: Vec<_> = indices
            .iter()
            .map(|&idx| {
                self.columns
                    .get(idx)
                    .ok_or_else(|| format!("Column index {} out of bounds", idx))
                    .map(|c| c.clone())
            })
            .collect::<Result<_, _>>()?;

        let schema = Arc::new(Schema::new(fields));

        Self::try_new(schema, columns)
    }

    /// Select a subset of columns by name
    /// 
    /// # Arguments
    /// * `names` - Vector of column names to select
    /// 
    /// # Returns
    /// A new RecordBatch containing only the selected columns
    pub fn select_columns_by_name(&self, names: &[&str]) -> Result<Self, String> {
        let indices: Vec<usize> = names
            .iter()
            .map(|name| {
                self.schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == *name)
                    .ok_or_else(|| format!("Column '{}' not found in schema", name))
            })
            .collect::<Result<_, _>>()?;

        self.select_columns(&indices)
    }

    /// Slice this batch to return a new batch with rows from `offset` to `offset + length`
    /// 
    /// # Arguments
    /// * `offset` - Starting row index
    /// * `length` - Number of rows to include
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self, String> {
        if offset + length > self.num_rows {
            return Err(format!(
                "Slice range [{}, {}) out of bounds for batch with {} rows",
                offset,
                offset + length,
                self.num_rows
            ));
        }

        let sliced_columns: Vec<ArrayRef> = self
            .columns
            .iter()
            .map(|col| col.slice(offset, length))
            .collect();

        Self::try_new(self.schema.clone(), sliced_columns)
    }

    /// Concatenate multiple RecordBatches together
    /// All batches must have the same schema
    pub fn concat(batches: &[Self]) -> Result<Self, String> {
        if batches.is_empty() {
            return Err("Cannot concatenate empty batch list".to_string());
        }

        // Verify all batches have the same schema
        let first_schema = batches[0].schema();
        for (idx, batch) in batches.iter().enumerate().skip(1) {
            if batch.schema() != first_schema {
                return Err(format!(
                    "Batch {} has different schema than first batch",
                    idx
                ));
            }
        }

        // Concatenate columns
        let num_columns = first_schema.fields().len();
        let mut concatenated_columns = Vec::with_capacity(num_columns);

        for col_idx in 0..num_columns {
            let column_arrays: Vec<ArrayRef> = batches
                .iter()
                .map(|batch| batch.columns[col_idx].clone())
                .collect();

            // Use Arrow's concat: it expects &[&dyn Array]
            let refs: Vec<&dyn arrow::array::Array> =
                column_arrays.iter().map(|a| a.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| format!("Failed to concatenate column {}: {}", col_idx, e))?;

            concatenated_columns.push(concatenated);
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows).sum();

        Self::try_new(first_schema.clone(), concatenated_columns).map(|batch| {
            // Verify the resulting batch has the expected number of rows
            debug_assert_eq!(batch.num_rows, total_rows);
            batch
        })
    }

    /// Check if the batch is empty (has zero rows)
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }
}

impl From<ArrowRecordBatch> for RecordBatch {
    fn from(batch: ArrowRecordBatch) -> Self {
        Self::from_arrow(batch)
    }
}

impl TryFrom<RecordBatch> for ArrowRecordBatch {
    type Error = String;

    fn try_from(batch: RecordBatch) -> Result<Self, Self::Error> {
        batch.to_arrow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Int32Array, StringArray};
    use arrow::datatypes::{Field, DataType};

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("active", DataType::Boolean, false),
        ]))
    }

    fn create_test_batch() -> RecordBatch {
        let schema = create_test_schema();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            Arc::new(BooleanArray::from(vec![true, false, true])),
        ];

        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn test_create_batch() {
        let batch = create_test_batch();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn test_column_access() {
        let batch = create_test_batch();
        
        // Test by index
        let col = batch.column(0).unwrap();
        assert_eq!(col.len(), 3);

        // Test by name
        let col = batch.column_by_name("id").unwrap();
        assert_eq!(col.len(), 3);

        // Test invalid index
        assert!(batch.column(10).is_err());
        
        // Test invalid name
        assert!(batch.column_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_select_columns() {
        let batch = create_test_batch();
        
        // Select first and third columns by index
        let selected = batch.select_columns(&[0, 2]).unwrap();
        assert_eq!(selected.num_columns(), 2);
        assert_eq!(selected.num_rows(), 3);

        // Select by name
        let selected = batch.select_columns_by_name(&["id", "name"]).unwrap();
        assert_eq!(selected.num_columns(), 2);
    }

    #[test]
    fn test_slice() {
        let batch = create_test_batch();
        
        let sliced = batch.slice(1, 2).unwrap();
        assert_eq!(sliced.num_rows(), 2);
        assert_eq!(sliced.num_columns(), 3);
    }

    #[test]
    fn test_concat() {
        let batch1 = create_test_batch();
        let batch2 = create_test_batch();
        
        let concatenated = RecordBatch::concat(&[batch1, batch2]).unwrap();
        assert_eq!(concatenated.num_rows(), 6);
        assert_eq!(concatenated.num_columns(), 3);
    }

    #[test]
    fn test_arrow_conversion() {
        let batch = create_test_batch();
        
        // Convert to Arrow and back
        let arrow_batch = batch.to_arrow().unwrap();
        let batch2 = RecordBatch::from_arrow(arrow_batch);
        
        assert_eq!(batch.num_rows(), batch2.num_rows());
        assert_eq!(batch.num_columns(), batch2.num_columns());
    }

    #[test]
    fn test_empty_batch() {
        let schema = create_test_schema();
        let empty_columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<String>::new())),
            Arc::new(BooleanArray::from(Vec::<bool>::new())),
        ];

        let batch = RecordBatch::try_new(schema, empty_columns).unwrap();
        assert!(batch.is_empty());
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_invalid_batch() {
        let schema = create_test_schema();
        
        // Wrong number of columns
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
        ];
        assert!(RecordBatch::try_new(schema.clone(), columns).is_err());

        // Inconsistent column lengths
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(BooleanArray::from(vec![true, false, true])),
        ];
        assert!(RecordBatch::try_new(schema, columns).is_err());
    }
}