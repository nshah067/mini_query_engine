// Vectorized filtering

use crate::execution::batch::{RecordBatch, SchemaRef};
use crate::execution::operators::Operator;
use crate::planner::logical_plan::{BinaryOp, LogicalExpr, LogicalValue};
use arrow::array::{ArrayRef, BooleanArray};
use arrow_ord::comparison::{eq_dyn, gt_dyn, gt_eq_dyn, lt_dyn, lt_eq_dyn, neq_dyn};
use std::sync::Arc;

/// Filter operator that applies a predicate expression to filter rows
/// Uses vectorized execution with Arrow's compute kernels
pub struct FilterOperator {
    predicate: LogicalExpr,
    schema: SchemaRef,
}

impl FilterOperator {
    /// Create a new Filter operator
    /// 
    /// # Arguments
    /// * `predicate` - Logical expression to use as the filter predicate
    /// * `input_schema` - Schema of the input data (needed to determine output schema)
    /// 
    /// # Returns
    /// Result containing the FilterOperator, or an error string
    pub fn new(predicate: LogicalExpr, input_schema: SchemaRef) -> Result<Self, String> {
        // Filter doesn't change the schema, so output schema is same as input
        Ok(Self {
            predicate,
            schema: input_schema,
        })
    }

    /// Evaluate a logical expression to a boolean array
    /// This is the core of vectorized expression evaluation
    fn evaluate_expr(
        &self,
        batch: &RecordBatch,
        expr: &LogicalExpr,
    ) -> Result<BooleanArray, String> {
        match expr {
            LogicalExpr::Column(_) => {
                Err("Cannot evaluate column as boolean without comparison".to_string())
            }
            LogicalExpr::Literal(LogicalValue::Boolean(value)) => {
                // Create a boolean array with all values set to the literal
                let len = batch.num_rows();
                Ok(BooleanArray::from(vec![*value; len]))
            }
            LogicalExpr::BinaryExpr { left, op, right } => {
                // Evaluate left and right sides to arrays
                let left_array = self.evaluate_to_array(batch, left)?;
                let right_array = self.evaluate_to_array(batch, right)?;

                // Apply binary operation using Arrow's vectorized compute (eq_dyn works with &dyn Array)
                match op {
                    BinaryOp::Eq => eq_dyn(left_array.as_ref(), right_array.as_ref())
                        .map_err(|e| format!("Failed to evaluate equality: {}", e)),
                    BinaryOp::Neq => neq_dyn(left_array.as_ref(), right_array.as_ref())
                        .map_err(|e| format!("Failed to evaluate inequality: {}", e)),
                    BinaryOp::Lt => lt_dyn(left_array.as_ref(), right_array.as_ref())
                        .map_err(|e| format!("Failed to evaluate less than: {}", e)),
                    BinaryOp::Le => lt_eq_dyn(left_array.as_ref(), right_array.as_ref())
                        .map_err(|e| format!("Failed to evaluate less than or equal: {}", e)),
                    BinaryOp::Gt => gt_dyn(left_array.as_ref(), right_array.as_ref())
                        .map_err(|e| format!("Failed to evaluate greater than: {}", e)),
                    BinaryOp::Ge => gt_eq_dyn(left_array.as_ref(), right_array.as_ref())
                        .map_err(|e| format!("Failed to evaluate greater than or equal: {}", e)),
                    BinaryOp::And => {
                        let left_bool = self.as_boolean_array(&left_array)?;
                        let right_bool = self.as_boolean_array(&right_array)?;
                        arrow::compute::and(left_bool, right_bool)
                            .map_err(|e| format!("Failed to evaluate AND: {}", e))
                    }
                    BinaryOp::Or => {
                        let left_bool = self.as_boolean_array(&left_array)?;
                        let right_bool = self.as_boolean_array(&right_array)?;
                        arrow::compute::or(left_bool, right_bool)
                            .map_err(|e| format!("Failed to evaluate OR: {}", e))
                    }
                }
            }
            LogicalExpr::Literal(LogicalValue::Int32(_))
            | LogicalExpr::Literal(LogicalValue::Int64(_))
            | LogicalExpr::Literal(LogicalValue::Float64(_))
            | LogicalExpr::Literal(LogicalValue::String(_)) => {
                Err("Non-boolean literal cannot be used as predicate".to_string())
            }
        }
    }

    /// Evaluate an expression to an Arrow array (not boolean)
    fn evaluate_to_array(
        &self,
        batch: &RecordBatch,
        expr: &LogicalExpr,
    ) -> Result<ArrayRef, String> {
        match expr {
            LogicalExpr::Column(name) => {
                batch
                    .column_by_name(name)
                    .ok_or_else(|| format!("Column '{}' not found", name))
                    .map(|col| col.clone())
            }
            LogicalExpr::Literal(value) => {
                let len = batch.num_rows();
                match value {
                    LogicalValue::Int32(v) => {
                        Ok(Arc::new(arrow::array::Int32Array::from(vec![*v; len])))
                    }
                    LogicalValue::Int64(v) => {
                        Ok(Arc::new(arrow::array::Int64Array::from(vec![*v; len])))
                    }
                    LogicalValue::Float64(v) => {
                        Ok(Arc::new(arrow::array::Float64Array::from(vec![*v; len])))
                    }
                    LogicalValue::String(v) => {
                        Ok(Arc::new(arrow::array::StringArray::from(vec![v.as_str(); len])))
                    }
                    LogicalValue::Boolean(v) => {
                        Ok(Arc::new(arrow::array::BooleanArray::from(vec![*v; len])))
                    }
                }
            }
            LogicalExpr::BinaryExpr { .. } => {
                // For binary expressions, evaluate to boolean first
                let bool_array = self.evaluate_expr(batch, expr)?;
                Ok(Arc::new(bool_array))
            }
        }
    }

    /// Convert an array to a boolean array reference
    fn as_boolean_array<'a>(&self, array: &'a ArrayRef) -> Result<&'a BooleanArray, String> {
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| "Array is not a boolean array".to_string())
    }
}

impl Operator for FilterOperator {
    /// Execute the filter operator on a batch
    /// Uses vectorized filtering with Arrow's compute kernels
    fn execute(&self, input: &RecordBatch) -> Result<RecordBatch, String> {
        // Evaluate the predicate to get a boolean mask
        let boolean_mask = self.evaluate_expr(input, &self.predicate)?;

        // Use Arrow's vectorized filter function to apply the mask to all columns
        // This is a vectorized operation processing the entire columns at once
        let filtered_columns: Vec<ArrayRef> = input
            .columns()
            .iter()
            .map(|col| {
                arrow::compute::filter(col, &boolean_mask)
                    .map_err(|e| format!("Failed to filter column: {}", e))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Create new batch with filtered columns
        RecordBatch::try_new(self.schema.clone(), filtered_columns)
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
