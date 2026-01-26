// DataFrame API implementation

use std::path::Path;

use crate::execution::batch::RecordBatch;
use crate::execution::Executor;
use crate::planner::logical_plan::{
    Aggregation, AggregateFunction, BinaryOp, JoinType, LogicalExpr, LogicalPlan, LogicalValue,
    OrderByExpr,
};

/// DataFrame represents a lazy query plan that can be executed
/// Operations on DataFrame build up a logical plan tree
#[derive(Debug, Clone)]
pub struct DataFrame {
    plan: LogicalPlan,
}

/// Intermediate type for group_by + agg. Call .agg(aggregations) to complete.
#[derive(Debug, Clone)]
pub struct GroupedDataFrame {
    input: LogicalPlan,
    group_by: Vec<String>,
}

impl GroupedDataFrame {
    /// Apply aggregations and return a DataFrame
    pub fn agg(self, aggs: Vec<Aggregation>) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Aggregate {
                input: Box::new(self.input),
                group_by: self.group_by,
                aggs,
            },
        }
    }
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

    /// Group by the given columns. Returns a GroupedDataFrame; call .agg(aggregations) to complete.
    pub fn group_by(&self, columns: Vec<String>) -> GroupedDataFrame {
        GroupedDataFrame {
            input: self.plan.clone(),
            group_by: columns,
        }
    }

    /// Order by the given expressions. Use `asc("col")` and `desc("col")` to build OrderByExpr.
    pub fn order_by(&self, order_by: Vec<OrderByExpr>) -> Self {
        DataFrame {
            plan: LogicalPlan::Sort {
                input: Box::new(self.plan.clone()),
                order_by,
            },
        }
    }

    /// Execute the query plan and return the results as a vector of RecordBatches
    /// 
    /// # Returns
    /// Vector of RecordBatches containing the query results
    pub fn collect(&self) -> Result<Vec<RecordBatch>, String> {
        Executor::new().execute(&self.plan)
    }
}

// Aggregation helper constructors for use with group_by().agg([...])
/// COUNT(*) - count all rows in each group
pub fn count(alias: &str) -> Aggregation {
    Aggregation {
        function: AggregateFunction::Count,
        column: None,
        alias: alias.to_string(),
    }
}

/// COUNT(column) - count non-null values in the column
pub fn count_column(column: &str, alias: &str) -> Aggregation {
    Aggregation {
        function: AggregateFunction::Count,
        column: Some(column.to_string()),
        alias: alias.to_string(),
    }
}

/// SUM(column)
pub fn sum(column: &str, alias: &str) -> Aggregation {
    Aggregation {
        function: AggregateFunction::Sum,
        column: Some(column.to_string()),
        alias: alias.to_string(),
    }
}

/// AVG(column)
pub fn avg(column: &str, alias: &str) -> Aggregation {
    Aggregation {
        function: AggregateFunction::Avg,
        column: Some(column.to_string()),
        alias: alias.to_string(),
    }
}

/// MIN(column)
pub fn min(column: &str, alias: &str) -> Aggregation {
    Aggregation {
        function: AggregateFunction::Min,
        column: Some(column.to_string()),
        alias: alias.to_string(),
    }
}

/// MAX(column)
pub fn max(column: &str, alias: &str) -> Aggregation {
    Aggregation {
        function: AggregateFunction::Max,
        column: Some(column.to_string()),
        alias: alias.to_string(),
    }
}

/// ORDER BY ascending
pub fn asc(column: &str) -> OrderByExpr {
    OrderByExpr {
        column: column.to_string(),
        ascending: true,
    }
}

/// ORDER BY descending
pub fn desc(column: &str) -> OrderByExpr {
    OrderByExpr {
        column: column.to_string(),
        ascending: false,
    }
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
