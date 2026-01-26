// ORDER BY sorting

use crate::execution::batch::{RecordBatch, SchemaRef};
use crate::execution::operators::Operator;
use crate::planner::logical_plan::OrderByExpr;
use arrow::array::ArrayRef;
use arrow_ord::sort::{lexsort_to_indices, SortColumn, SortOptions};
use arrow_select::take::take;

/// Sort operator for ORDER BY
/// Uses arrow_ord::lexsort for lexicographic multi-column sort
pub struct SortOperator {
    order_by: Vec<OrderByExpr>,
    schema: SchemaRef,
}

impl SortOperator {
    /// Create a new Sort operator
    pub fn new(order_by: Vec<OrderByExpr>, input_schema: SchemaRef) -> Result<Self, String> {
        // Validate that all order_by columns exist
        for e in &order_by {
            input_schema
                .fields()
                .iter()
                .find(|f| f.name() == e.column.as_str())
                .ok_or_else(|| format!("Order column '{}' not found", e.column))?;
        }
        Ok(Self {
            order_by,
            schema: input_schema,
        })
    }

    /// Sort a single batch
    fn sort_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, String> {
        if batch.num_rows() == 0 {
            return Ok(batch.clone());
        }
        if self.order_by.is_empty() {
            return Ok(batch.clone());
        }

        let sort_columns: Vec<SortColumn> = self
            .order_by
            .iter()
            .map(|e| {
                let col = batch
                    .column_by_name(&e.column)
                    .ok_or_else(|| format!("Column '{}' not found", e.column))
                    .map(|c| c.clone())?;
                Ok(SortColumn {
                    values: col,
                    options: Some(SortOptions {
                        descending: !e.ascending,
                        nulls_first: true,
                    }),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let indices = lexsort_to_indices(&sort_columns, None)
            .map_err(|e| format!("Sort failed: {}", e))?;

        // Apply take to each column in the batch
        let sorted_columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| take(col.as_ref(), &indices, None).map_err(|e| format!("Take failed: {}", e)))
            .collect::<Result<Vec<_>, _>>()?;

        RecordBatch::try_new(self.schema.clone(), sorted_columns)
    }
}

impl Operator for SortOperator {
    fn execute(&self, input: &RecordBatch) -> Result<RecordBatch, String> {
        self.sort_batch(input)
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute_many(&self, inputs: &[RecordBatch]) -> Result<Vec<RecordBatch>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        // Concat all batches then sort (for correct global ORDER BY)
        let combined = RecordBatch::concat(inputs)?;
        let sorted = self.sort_batch(&combined)?;
        Ok(if sorted.is_empty() { vec![] } else { vec![sorted] })
    }
}
