pub mod aggregate;
pub mod filter;
pub mod join;
pub mod project;
pub mod scan;
pub mod sort;

// Export operators for use by executor
pub use filter::FilterOperator;
pub use project::ProjectOperator;
pub use scan::ScanOperator;

use crate::execution::batch::{RecordBatch, SchemaRef};
use std::sync::Arc;

/// Trait for all execution operators in the query engine
/// Operators process RecordBatches in a vectorized manner
pub trait Operator: Send + Sync {
    /// Execute the operator on a batch of data
    /// 
    /// # Arguments
    /// * `input` - Input RecordBatch to process
    /// 
    /// # Returns
    /// Result containing the output RecordBatch, or an error string
    fn execute(&self, input: &RecordBatch) -> Result<RecordBatch, String>;

    /// Get the output schema of this operator
    /// 
    /// # Returns
    /// The schema that this operator will produce
    fn schema(&self) -> SchemaRef;

    /// Execute the operator on multiple batches (for operators that can process multiple inputs)
    /// Default implementation processes each batch individually
    /// 
    /// # Arguments
    /// * `inputs` - Vector of input RecordBatches
    /// 
    /// # Returns
    /// Result containing vector of output RecordBatches
    fn execute_many(&self, inputs: &[RecordBatch]) -> Result<Vec<RecordBatch>, String> {
        inputs.iter().map(|batch| self.execute(batch)).collect()
    }
}
