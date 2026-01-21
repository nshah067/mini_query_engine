// Column selection/projection

use crate::execution::batch::{RecordBatch, SchemaRef};
use crate::execution::operators::Operator;
use arrow::datatypes::{Field, Schema};
use std::sync::Arc;

/// Project operator that selects a subset of columns
/// Uses vectorized column selection for efficient projection
pub struct ProjectOperator {
    column_names: Vec<String>,
    column_indices: Vec<usize>,
    schema: SchemaRef,
}

impl ProjectOperator {
    /// Create a new Project operator
    /// 
    /// # Arguments
    /// * `column_names` - Names of columns to select
    /// * `input_schema` - Schema of the input data
    /// 
    /// # Returns
    /// Result containing the ProjectOperator, or an error string
    pub fn new(column_names: Vec<String>, input_schema: SchemaRef) -> Result<Self, String> {
        // Find column indices and build output schema
        let mut column_indices = Vec::with_capacity(column_names.len());
        let mut fields = Vec::with_capacity(column_names.len());

        for name in &column_names {
            let (idx, field) = input_schema
                .fields()
                .iter()
                .enumerate()
                .find(|(_, f)| f.name() == name)
                .ok_or_else(|| format!("Column '{}' not found in schema", name))?;
            
            column_indices.push(idx);
            fields.push(field.clone());
        }

        let schema = Arc::new(Schema::new(fields));

        Ok(Self {
            column_names,
            column_indices,
            schema,
        })
    }
}

impl Operator for ProjectOperator {
    /// Execute the project operator on a batch
    /// Uses vectorized column selection
    fn execute(&self, input: &RecordBatch) -> Result<RecordBatch, String> {
        // Use the batch's select_columns method which is already vectorized
        input.select_columns(&self.column_indices)
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
