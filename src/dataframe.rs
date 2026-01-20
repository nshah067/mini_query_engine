// DataFrame API implementation

use std::path::Path;

use crate::execution::batch::RecordBatch;
use crate::planner::logical_plan::{BinaryOp, LogicalExpr, LogicalPlan, LogicalValue};
use crate::storage::parquet_reader;

/// DataFrame represents a lazy query plan that can be executed
/// Operations on DataFrame build up a logical plan tree
#[derive(Debug, Clone)]
pub struct DataFrame {
    plan: LogicalPlan,
}

impl DataFrame {
    /// Create a DataFrame from a Parquet file path
    /// 
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// 
    /// # Returns
    /// A new DataFrame with a Scan operation in the plan
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_buf = path.as_ref().to_path_buf();
        Ok(DataFrame {
            plan: LogicalPlan::Scan {
                path: path_buf,
                projection: None,
                filters: vec![],
            },
        })
    }

    /// Select specific columns (projection)
    /// 
    /// # Arguments
    /// * `columns` - Vector of column names to select
    /// 
    /// # Returns
    /// A new DataFrame with a Project operation added to the plan
    pub fn select(&self, columns: Vec<String>) -> Self {
        DataFrame {
            plan: LogicalPlan::Project {
                input: Box::new(self.plan.clone()),
                columns,
            },
        }
    }

    /// Filter rows based on a predicate expression
    /// 
    /// # Arguments
    /// * `predicate` - A logical expression to use as a filter predicate
    /// 
    /// # Example
    /// ```ignore
    /// use mini_query_engine::dataframe::{col, lit_int32};
    /// df.filter(col("age").gt(lit_int32(18)))
    /// ```
    pub fn filter(&self, predicate: LogicalExpr) -> Self {
        DataFrame {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan.clone()),
                predicate,
            },
        }
    }

    /// Execute the query plan and return the results as a vector of RecordBatches
    /// 
    /// # Returns
    /// Vector of RecordBatches containing the query results
    pub fn collect(&self) -> Result<Vec<RecordBatch>, String> {
        execute_plan(&self.plan)
    }
}

// Helper function to execute a logical plan
fn execute_plan(plan: &LogicalPlan) -> Result<Vec<RecordBatch>, String> {
    match plan {
        LogicalPlan::Scan { path, projection, filters } => {
            // Read from Parquet file
            let mut reader = parquet_reader::ParquetReader::from_path(path)
                .map_err(|e| format!("Failed to read Parquet file: {}", e))?;
            
            let arrow_batches = reader.read_all()
                .map_err(|e| format!("Failed to read all data: {}", e))?;

            // Convert Arrow RecordBatches to our RecordBatch type
            let batches: Vec<RecordBatch> = arrow_batches
                .into_iter()
                .map(RecordBatch::from_arrow)
                .collect();

            // Apply column projection if specified
            let batches = if let Some(columns) = projection {
                let indices: Result<Vec<usize>, String> = batches
                    .first()
                    .ok_or("No batches to project")?
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, field)| {
                        if columns.contains(&field.name().to_string()) {
                            Some(Ok(idx))
                        } else {
                            None
                        }
                    })
                    .collect();
                let indices = indices?;
                batches
                    .into_iter()
                    .map(|batch| batch.select_columns(&indices))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                batches
            };

            // Apply filters if specified
            // For now, filters will be applied during execution
            // This is a simplified implementation
            if !filters.is_empty() {
                // Filters will be applied by Filter operators in the plan
                // For Scan-level filters, we could push them down here
            }

            Ok(batches)
        }
        LogicalPlan::Project { input, columns } => {
            // Execute input first
            let input_batches = execute_plan(input)?;

            // Get column indices
            let indices: Vec<usize> = input_batches
                .first()
                .ok_or("No batches to project")?
                .schema()
                .fields()
                .iter()
                .enumerate()
                .filter_map(|(idx, field)| {
                    if columns.contains(&field.name().to_string()) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            if indices.len() != columns.len() {
                return Err(format!(
                    "Some columns not found. Requested: {:?}, Found: {:?}",
                    columns,
                    input_batches[0]
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().to_string())
                        .collect::<Vec<_>>()
                ));
            }

            // Apply projection to each batch
            let projected_batches: Result<Vec<RecordBatch>, String> = input_batches
                .into_iter()
                .map(|batch| batch.select_columns(&indices))
                .collect();

            projected_batches
        }
        LogicalPlan::Filter { input, predicate } => {
            // Execute input first
            let input_batches = execute_plan(input)?;

            // Apply filter to each batch
            let filtered_batches: Result<Vec<RecordBatch>, String> = input_batches
                .into_iter()
                .map(|batch| apply_filter(&batch, predicate))
                .collect();

            // Filter out empty batches
            let filtered_batches: Vec<RecordBatch> = filtered_batches?
                .into_iter()
                .filter(|b| !b.is_empty())
                .collect();

            Ok(filtered_batches)
        }
    }
}

// Helper function to apply a filter expression to a RecordBatch
fn apply_filter(batch: &RecordBatch, predicate: &LogicalExpr) -> Result<RecordBatch, String> {
    // Evaluate the predicate to get a boolean array
    let boolean_array = evaluate_expr(batch, predicate)?;

    // Use Arrow's filter function to apply the boolean mask
    let filtered_columns: Vec<arrow::array::ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| {
            arrow::compute::filter(col, &boolean_array)
                .map_err(|e| format!("Failed to filter column: {}", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let num_rows = filtered_columns.first().map(|c| c.len()).unwrap_or(0);
    RecordBatch::try_new(batch.schema().clone(), filtered_columns)
}

// Helper function to evaluate a logical expression against a RecordBatch
fn evaluate_expr(
    batch: &RecordBatch,
    expr: &LogicalExpr,
) -> Result<arrow::array::BooleanArray, String> {
    match expr {
        LogicalExpr::Column(name) => {
            // For now, we can't evaluate a column as a boolean without a comparison
            // This is a limitation - in a full implementation, we'd handle this differently
            Err(format!(
                "Cannot evaluate column '{}' as boolean expression",
                name
            ))
        }
        LogicalExpr::Literal(LogicalValue::Boolean(value)) => {
            // Create a boolean array with all values set to the literal
            let len = batch.num_rows();
            Ok(arrow::array::BooleanArray::from(vec![*value; len]))
        }
        LogicalExpr::BinaryExpr { left, op, right } => {
            // Evaluate left and right sides
            let left_array = evaluate_to_array(batch, left)?;
            let right_array = evaluate_to_array(batch, right)?;

            // Apply binary operation
            match op {
                BinaryOp::Eq => {
                    arrow::compute::eq(&left_array, &right_array)
                        .map_err(|e| format!("Failed to evaluate equality: {}", e))
                }
                BinaryOp::Neq => {
                    arrow::compute::neq(&left_array, &right_array)
                        .map_err(|e| format!("Failed to evaluate inequality: {}", e))
                }
                BinaryOp::Lt => {
                    arrow::compute::lt(&left_array, &right_array)
                        .map_err(|e| format!("Failed to evaluate less than: {}", e))
                }
                BinaryOp::Le => {
                    arrow::compute::lt_eq(&left_array, &right_array)
                        .map_err(|e| format!("Failed to evaluate less than or equal: {}", e))
                }
                BinaryOp::Gt => {
                    arrow::compute::gt(&left_array, &right_array)
                        .map_err(|e| format!("Failed to evaluate greater than: {}", e))
                }
                BinaryOp::Ge => {
                    arrow::compute::gt_eq(&left_array, &right_array)
                        .map_err(|e| format!("Failed to evaluate greater than or equal: {}", e))
                }
                BinaryOp::And => {
                    let left_bool = as_boolean_array(&left_array)?;
                    let right_bool = as_boolean_array(&right_array)?;
                    arrow::compute::and(&left_bool, &right_bool)
                        .map_err(|e| format!("Failed to evaluate AND: {}", e))
                }
                BinaryOp::Or => {
                    let left_bool = as_boolean_array(&left_array)?;
                    let right_bool = as_boolean_array(&right_array)?;
                    arrow::compute::or(&left_bool, &right_bool)
                        .map_err(|e| format!("Failed to evaluate OR: {}", e))
                }
            }
        }
        _ => Err(format!("Unsupported expression type: {:?}", expr)),
    }
}

// Helper to evaluate an expression to an Arrow array
fn evaluate_to_array(
    batch: &RecordBatch,
    expr: &LogicalExpr,
) -> Result<arrow::array::ArrayRef, String> {
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
            // For binary expressions, we evaluate them to boolean first, then convert
            let bool_array = evaluate_expr(batch, expr)?;
            Ok(Arc::new(bool_array))
        }
    }
}

// Helper to convert an array to a boolean array
fn as_boolean_array(array: &arrow::array::ArrayRef) -> Result<&arrow::array::BooleanArray, String> {
    array
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .ok_or_else(|| "Array is not a boolean array".to_string())
}

// Helper functions for building expressions more easily
// These can be used with the filter method

/// Helper to create a column reference expression
pub fn col(name: &str) -> LogicalExpr {
    LogicalExpr::Column(name.to_string())
}

/// Extension trait for building expressions
pub trait ExprBuilder {
    fn eq(&self, other: LogicalExpr) -> LogicalExpr;
    fn neq(&self, other: LogicalExpr) -> LogicalExpr;
    fn gt(&self, other: LogicalExpr) -> LogicalExpr;
    fn ge(&self, other: LogicalExpr) -> LogicalExpr;
    fn lt(&self, other: LogicalExpr) -> LogicalExpr;
    fn le(&self, other: LogicalExpr) -> LogicalExpr;
}

impl ExprBuilder for LogicalExpr {
    fn eq(&self, other: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self.clone()),
            op: BinaryOp::Eq,
            right: Box::new(other),
        }
    }

    fn neq(&self, other: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self.clone()),
            op: BinaryOp::Neq,
            right: Box::new(other),
        }
    }

    fn gt(&self, other: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self.clone()),
            op: BinaryOp::Gt,
            right: Box::new(other),
        }
    }

    fn ge(&self, other: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self.clone()),
            op: BinaryOp::Ge,
            right: Box::new(other),
        }
    }

    fn lt(&self, other: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self.clone()),
            op: BinaryOp::Lt,
            right: Box::new(other),
        }
    }

    fn le(&self, other: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self.clone()),
            op: BinaryOp::Le,
            right: Box::new(other),
        }
    }
}

// Helper functions for literals
pub fn lit_int32(v: i32) -> LogicalExpr {
    LogicalExpr::Literal(LogicalValue::Int32(v))
}

pub fn lit_int64(v: i64) -> LogicalExpr {
    LogicalExpr::Literal(LogicalValue::Int64(v))
}

pub fn lit_float64(v: f64) -> LogicalExpr {
    LogicalExpr::Literal(LogicalValue::Float64(v))
}

pub fn lit_string(v: &str) -> LogicalExpr {
    LogicalExpr::Literal(LogicalValue::String(v.to_string()))
}

pub fn lit_bool(v: bool) -> LogicalExpr {
    LogicalExpr::Literal(LogicalValue::Boolean(v))
}
