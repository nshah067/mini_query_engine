// Logical query plan

use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;

/// Logical expression for filtering
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    /// Column reference by name
    Column(String),
    /// Literal value
    Literal(LogicalValue),
    /// Binary comparison: left op right
    BinaryExpr {
        left: Box<LogicalExpr>,
        op: BinaryOp,
        right: Box<LogicalExpr>,
    },
}

/// Binary operators for expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,   // ==
    Neq,  // !=
    Lt,   // <
    Le,   // <=
    Gt,   // >
    Ge,   // >=
    And,  // &&
    Or,   // ||
}

/// Literal values in expressions
#[derive(Debug, Clone)]
pub enum LogicalValue {
    Int32(i32),
    Int64(i64),
    Float64(f64),
    String(String),
    Boolean(bool),
}

/// Aggregate function for GROUP BY aggregations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// An aggregation expression: function, optional column (None for Count(*)), and output alias
#[derive(Debug, Clone)]
pub struct Aggregation {
    pub function: AggregateFunction,
    pub column: Option<String>,
    pub alias: String,
}

/// Logical query plan representing a query as a tree of operations
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a Parquet file
    Scan {
        path: PathBuf,
        projection: Option<Vec<String>>, // Column names to read
        filters: Vec<LogicalExpr>,       // Predicate pushdown filters
    },
    /// Select/project specific columns
    Project {
        input: Box<LogicalPlan>,
        columns: Vec<String>, // Column names to select
    },
    /// Filter rows based on a predicate
    Filter {
        input: Box<LogicalPlan>,
        predicate: LogicalExpr,
    },
    /// Aggregate with GROUP BY
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<String>,
        aggs: Vec<Aggregation>,
    },
    /// ORDER BY
    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<OrderByExpr>,
    },
    /// Join two plans
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        on: (String, String), // (left_key, right_key)
    },
}

/// Join type: Inner or Left (outer)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
}

/// Expression for ORDER BY: column name and direction
#[derive(Debug, Clone)]
pub struct OrderByExpr {
    pub column: String,
    pub ascending: bool,
}

impl LogicalPlan {
    /// Get the output schema for this plan node
    pub fn schema(&self) -> Result<SchemaRef, String> {
        match self {
            LogicalPlan::Scan { .. } => {
                // For scan, we need to read the schema from the file
                // This will be handled during execution
                Err("Schema not available for Scan without execution".to_string())
            }
            LogicalPlan::Project { input, columns } => {
                let input_schema = input.schema()?;
                let fields: Vec<_> = columns
                    .iter()
                    .map(|name| {
                        input_schema
                            .fields()
                            .iter()
                            .find(|f| f.name() == name)
                            .ok_or_else(|| format!("Column '{}' not found in schema", name))
                            .map(|f| f.clone())
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Arc::new(arrow::datatypes::Schema::new(fields)))
            }
            LogicalPlan::Filter { input, .. } => {
                // Filter doesn't change schema
                input.schema()
            }
            LogicalPlan::Aggregate { .. } => {
                // Schema is computed during execution based on group_by + aggs
                Err("Schema not available for Aggregate without execution".to_string())
            }
            LogicalPlan::Sort { input, .. } => {
                // Sort doesn't change schema
                input.schema()
            }
            LogicalPlan::Join { .. } => {
                Err("Schema not available for Join without execution".to_string())
            }
        }
    }
}
