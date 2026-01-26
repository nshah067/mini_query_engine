// Execution engine coordinator

use crate::execution::batch::RecordBatch;
use crate::execution::operators::{
    AggregateOperator, FilterOperator, HashJoinOperator, Operator, ProjectOperator, ScanOperator,
    SortOperator,
};
use crate::planner::logical_plan::{AggregateFunction, JoinType, LogicalPlan};
use crate::storage::parquet_reader::ParquetReader;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Executor that coordinates the execution of logical plans
/// Converts logical plans to physical operators and executes them
pub struct Executor;

impl Executor {
    /// Create a new executor
    pub fn new() -> Self {
        Self
    }

    /// Execute a logical plan and return the results
    /// 
    /// # Arguments
    /// * `plan` - The logical plan to execute
    /// 
    /// # Returns
    /// Result containing vector of RecordBatches with the query results
    pub fn execute(&self, plan: &LogicalPlan) -> Result<Vec<RecordBatch>, String> {
        match plan {
            LogicalPlan::Scan { path, projection, .. } => {
                // Create and execute Scan operator
                let scan_op = ScanOperator::new(path, projection.clone())?;
                scan_op.read_all()
            }
            LogicalPlan::Project { input, columns } => {
                // Execute input first
                let input_batches = self.execute(input)?;
                
                if input_batches.is_empty() {
                    return Ok(Vec::new());
                }

                // Create Project operator using the input schema
                let input_schema = input_batches[0].schema().clone();
                let project_op = ProjectOperator::new(columns.clone(), input_schema)?;

                // Apply projection to each batch
                let projected_batches: Result<Vec<RecordBatch>, String> = input_batches
                    .iter()
                    .map(|batch| project_op.execute(batch))
                    .collect();

                projected_batches
            }
            LogicalPlan::Filter { input, predicate } => {
                // Execute input first
                let input_batches = self.execute(input)?;
                
                if input_batches.is_empty() {
                    return Ok(Vec::new());
                }

                // Create Filter operator using the input schema
                let input_schema = input_batches[0].schema().clone();
                let filter_op = FilterOperator::new(predicate.clone(), input_schema)?;

                // Apply filter to each batch
                let filtered_batches: Result<Vec<RecordBatch>, String> = input_batches
                    .iter()
                    .map(|batch| filter_op.execute(batch))
                    .collect();

                // Filter out empty batches
                let filtered_batches: Vec<RecordBatch> = filtered_batches?
                    .into_iter()
                    .filter(|b| !b.is_empty())
                    .collect();

                Ok(filtered_batches)
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggs,
            } => {
                let input_batches = self.execute(input)?;
                if input_batches.is_empty() {
                    // Build empty result with correct output schema (placeholder types for group cols)
                    let mut fields: Vec<Field> = group_by
                        .iter()
                        .map(|n| Field::new(n, DataType::Utf8, true))
                        .collect();
                    for a in aggs {
                        let dt = match a.function {
                            AggregateFunction::Count => DataType::Int64,
                            _ => DataType::Float64,
                        };
                        fields.push(Field::new(a.alias.as_str(), dt, true));
                    }
                    let schema = Arc::new(Schema::new(fields));
                    let columns: Vec<_> = schema
                        .fields()
                        .iter()
                        .map(|f| arrow::array::new_empty_array(f.data_type()))
                        .collect();
                    let batch = RecordBatch::try_new(schema, columns)
                        .map_err(|e| e.to_string())?;
                    return Ok(vec![batch]);
                }
                let input_schema = input_batches[0].schema().clone();
                let agg_op =
                    AggregateOperator::new(group_by.clone(), aggs.clone(), input_schema)
                        .map_err(|e| e.to_string())?;
                agg_op.execute_many(&input_batches)
            }
            LogicalPlan::Sort { input, order_by } => {
                let input_batches = self.execute(input)?;
                if input_batches.is_empty() {
                    return Ok(Vec::new());
                }
                let input_schema = input_batches[0].schema().clone();
                let sort_op = SortOperator::new(order_by.clone(), input_schema)
                    .map_err(|e| e.to_string())?;
                sort_op.execute_many(&input_batches)
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                on: (left_key, right_key),
            } => {
                let left_batches = self.execute(left)?;
                let right_batches = self.execute(right)?;

                if left_batches.is_empty() {
                    return Ok(Vec::new());
                }
                let left_schema = left_batches[0].schema().clone();
                let right_schema = right_batches
                    .first()
                    .map(|b| b.schema().clone())
                    .or_else(|| self.get_schema(right).ok())
                    .ok_or("Join right side has no batches and schema could not be determined")?;

                let join_op = HashJoinOperator::new(
                    left_key.clone(),
                    right_key.clone(),
                    *join_type,
                    left_schema,
                    right_schema,
                )
                .map_err(|e| e.to_string())?;
                join_op.execute_join(&left_batches, &right_batches)
            }
        }
    }

    /// Get the output schema of a plan without fully executing it (e.g. for Scan, read metadata only).
    fn get_schema(&self, plan: &LogicalPlan) -> Result<SchemaRef, String> {
        match plan {
            LogicalPlan::Scan { path, projection, .. } => {
                let s = ParquetReader::from_path(path)
                    .map_err(|e| e.to_string())?
                    .schema()
                    .map_err(|e| e.to_string())?;
                let schema = if let Some(ref cols) = projection {
                    let fields: Vec<Field> = cols
                        .iter()
                        .map(|n| {
                            s.fields()
                                .iter()
                                .find(|f| f.name().as_ref() == n.as_str())
                                .ok_or_else(|| format!("Column '{}' not found", n))
                                .map(|f| f.as_ref().clone())
                        })
                        .collect::<Result<_, _>>()?;
                    Arc::new(Schema::new(fields))
                } else {
                    Arc::new(s)
                };
                Ok(schema)
            }
            LogicalPlan::Project { input, columns } => {
                let in_s = self.get_schema(input)?;
                let fields: Vec<Field> = columns
                    .iter()
                    .map(|n| {
                        in_s
                            .fields()
                            .iter()
                            .find(|f| f.name().as_ref() == n.as_str())
                            .ok_or_else(|| format!("Column '{}' not found", n))
                            .map(|f| f.as_ref().clone())
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Arc::new(Schema::new(fields)))
            }
            LogicalPlan::Filter { input, .. } | LogicalPlan::Sort { input, .. } => self.get_schema(input),
            LogicalPlan::Aggregate { .. } | LogicalPlan::Join { .. } => {
                Err("get_schema not supported for Aggregate/Join".to_string())
            }
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}
