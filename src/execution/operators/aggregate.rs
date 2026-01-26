// GROUP BY aggregations

use crate::execution::batch::{RecordBatch, SchemaRef};
use crate::execution::operators::Operator;
use crate::planner::logical_plan::{AggregateFunction, Aggregation};
use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

/// Scalar value for group keys - supports types we need for GROUP BY
#[derive(Clone, Debug)]
enum GroupValue {
    I32(i32),
    I64(i64),
    F64(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl GroupValue {
    fn to_key_string(&self) -> String {
        match self {
            GroupValue::I32(v) => format!("i32:{}", v),
            GroupValue::I64(v) => format!("i64:{}", v),
            GroupValue::F64(v) => format!("f64:{}", v),
            GroupValue::Str(v) => format!("str:{}", v),
            GroupValue::Bool(v) => format!("bool:{}", v),
            GroupValue::Null => "null".to_string(),
        }
    }
}

/// Per-aggregation state
#[derive(Clone, Debug)]
enum AggState {
    Count(u64),
    Sum(f64),
    Avg { sum: f64, count: u64 },
    Min(f64),
    Max(f64),
}

/// Aggregate operator implementing GROUP BY with COUNT, SUM, AVG, MIN, MAX
/// Uses vectorized hash aggregation: builds a hash map of group key -> aggregate states
pub struct AggregateOperator {
    group_by: Vec<String>,
    aggs: Vec<Aggregation>,
    schema: SchemaRef,
}

impl AggregateOperator {
    /// Create a new Aggregate operator
    pub fn new(
        group_by: Vec<String>,
        aggs: Vec<Aggregation>,
        input_schema: SchemaRef,
    ) -> Result<Self, String> {
        // Build output schema: group_by columns + agg result columns
        let mut fields: Vec<Field> = Vec::new();

        for name in &group_by {
            let field = input_schema
                .fields()
                .iter()
                .find(|f| f.name() == name)
                .ok_or_else(|| format!("Group column '{}' not found", name))?
                .as_ref()
                .clone();
            fields.push(field);
        }

        for agg in &aggs {
            let data_type = match agg.function {
                AggregateFunction::Count => DataType::Int64,
                AggregateFunction::Sum | AggregateFunction::Avg | AggregateFunction::Min
                | AggregateFunction::Max => DataType::Float64,
            };
            fields.push(Field::new(agg.alias.as_str(), data_type, true));
        }

        let schema = Arc::new(Schema::new(fields));

        Ok(Self {
            group_by,
            aggs,
            schema,
        })
    }

    /// Extract group key from a row as string (for hashing)
    fn get_group_key(&self, batch: &RecordBatch, row: usize) -> Result<String, String> {
        let mut parts = Vec::with_capacity(self.group_by.len());
        for name in &self.group_by {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| format!("Column '{}' not found", name))?;
            let gv = extract_group_value(col, row)?;
            parts.push(gv.to_key_string());
        }
        Ok(parts.join("|"))
    }

    /// Extract group values from a row (for output)
    fn get_group_values(&self, batch: &RecordBatch, row: usize) -> Result<Vec<GroupValue>, String> {
        self.group_by
            .iter()
            .map(|name| {
                let col = batch
                    .column_by_name(name)
                    .ok_or_else(|| format!("Column '{}' not found", name))?;
                extract_group_value(col, row)
            })
            .collect()
    }

    /// Get numeric value from column for aggregations
    fn get_agg_value(&self, batch: &RecordBatch, agg: &Aggregation, row: usize) -> Option<f64> {
        let col = if let Some(ref c) = agg.column {
            batch.column_by_name(c)?
        } else {
            return None; // Count(*) doesn't need a column value
        };
        extract_numeric(col, row)
    }

    /// Process all batches and produce one aggregated batch
    fn hash_aggregate(&self, inputs: &[RecordBatch]) -> Result<RecordBatch, String> {
        // Map: group_key_string -> (group_values, agg_states)
        // We keep group_values from first occurrence for output
        let mut map: HashMap<String, (Vec<GroupValue>, Vec<AggState>)> = HashMap::new();

        for batch in inputs {
            if batch.num_rows() == 0 {
                continue;
            }

            for row in 0..batch.num_rows() {
                let key = self.get_group_key(batch, row)?;
                let group_vals = self.get_group_values(batch, row)?;

                let entry = map
                    .entry(key)
                    .or_insert_with(|| (group_vals.clone(), self.initial_states()));

                let states = &mut entry.1;

                for (i, agg) in self.aggs.iter().enumerate() {
                    match agg.function {
                        AggregateFunction::Count => {
                            let v = if agg.column.is_none() {
                                1.0
                            } else {
                                match self.get_agg_value(batch, agg, row) {
                                    Some(_) => 1.0,
                                    None => 0.0, // null doesn't count for count(col)
                                }
                            };
                            if let AggState::Count(ref mut c) = states[i] {
                                *c += if v > 0.0 { 1 } else { 0 };
                            }
                        }
                        AggregateFunction::Sum => {
                            if let Some(v) = self.get_agg_value(batch, agg, row) {
                                if let AggState::Sum(ref mut s) = states[i] {
                                    *s += v;
                                }
                            }
                        }
                        AggregateFunction::Avg => {
                            if let Some(v) = self.get_agg_value(batch, agg, row) {
                                if let AggState::Avg { sum, count } = &mut states[i] {
                                    *sum += v;
                                    *count += 1;
                                }
                            }
                        }
                        AggregateFunction::Min => {
                            if let Some(v) = self.get_agg_value(batch, agg, row) {
                                if let AggState::Min(ref mut m) = states[i] {
                                    if *m > v {
                                        *m = v;
                                    }
                                }
                            }
                        }
                        AggregateFunction::Max => {
                            if let Some(v) = self.get_agg_value(batch, agg, row) {
                                if let AggState::Max(ref mut m) = states[i] {
                                    if *m < v {
                                        *m = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.build_output_batch(map)
    }

    fn initial_states(&self) -> Vec<AggState> {
        self.aggs
            .iter()
            .map(|a| match a.function {
                AggregateFunction::Count => AggState::Count(0),
                AggregateFunction::Sum => AggState::Sum(0.0),
                AggregateFunction::Avg => AggState::Avg { sum: 0.0, count: 0 },
                AggregateFunction::Min => AggState::Min(f64::INFINITY),
                AggregateFunction::Max => AggState::Max(f64::NEG_INFINITY),
            })
            .collect()
    }

    fn build_output_batch(
        &self,
        map: HashMap<String, (Vec<GroupValue>, Vec<AggState>)>,
    ) -> Result<RecordBatch, String> {
        let n = map.len();
        if n == 0 {
            let empty_cols: Vec<ArrayRef> = self
                .schema
                .fields()
                .iter()
                .map(|f| arrow::array::new_empty_array(f.data_type()))
                .collect();
            return RecordBatch::try_new(self.schema.clone(), empty_cols);
        }

        // Build column arrays: first group_by columns, then agg columns
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.schema.fields().len());

        let num_group = self.group_by.len();
        let num_aggs = self.aggs.len();

        // For each group column, collect values (use schema for type when all nulls)
        for g in 0..num_group {
            let dt = self.schema.fields()[g].data_type().clone();
            let arr = collect_group_column(
                map.values().map(|(vals, _)| &vals[g]),
                &dt,
            )?;
            columns.push(arr);
        }

        // For each agg, collect final values
        for a in 0..num_aggs {
            let arr = collect_agg_column(
                &self.aggs[a],
                map.values().map(|(_, sts)| &sts[a]),
            )?;
            columns.push(arr);
        }

        RecordBatch::try_new(self.schema.clone(), columns)
    }
}

fn extract_group_value(col: &ArrayRef, row: usize) -> Result<GroupValue, String> {
    use arrow::array::*;
    if col.is_null(row) {
        return Ok(GroupValue::Null);
    }
    match col.data_type() {
        DataType::Int32 => {
            let arr = col.as_any().downcast_ref::<Int32Array>().ok_or("Int32")?;
            Ok(GroupValue::I32(arr.value(row)))
        }
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().ok_or("Int64")?;
            Ok(GroupValue::I64(arr.value(row)))
        }
        DataType::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>().ok_or("Float64")?;
            Ok(GroupValue::F64(arr.value(row)))
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().ok_or("Utf8")?;
            Ok(GroupValue::Str(arr.value(row).to_string()))
        }
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().ok_or("Boolean")?;
            Ok(GroupValue::Bool(arr.value(row)))
        }
        _ => Err(format!("Unsupported group type: {:?}", col.data_type())),
    }
}

fn extract_numeric(col: &ArrayRef, row: usize) -> Option<f64> {
    use arrow::array::*;
    if col.is_null(row) {
        return None;
    }
    match col.data_type() {
        DataType::Int32 => {
            let arr = col.as_any().downcast_ref::<Int32Array>()?;
            Some(arr.value(row) as f64)
        }
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>()?;
            Some(arr.value(row) as f64)
        }
        DataType::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>()?;
            Some(arr.value(row))
        }
        _ => None,
    }
}

fn collect_group_column<'a, I>(it: I, default_type: &DataType) -> Result<ArrayRef, String>
where
    I: Iterator<Item = &'a GroupValue>,
{
    let vec: Vec<&GroupValue> = it.collect();
    if vec.is_empty() {
        return Err("empty".to_string());
    }
    let first = vec[0];
    match first {
        GroupValue::I32(_) => {
            let arr: Vec<Option<i32>> = vec
                .iter()
                .map(|v| {
                    if let GroupValue::I32(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Int32Array::from(arr)) as ArrayRef)
        }
        GroupValue::I64(_) => {
            let arr: Vec<Option<i64>> = vec
                .iter()
                .map(|v| {
                    if let GroupValue::I64(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Int64Array::from(arr)) as ArrayRef)
        }
        GroupValue::F64(_) => {
            let arr: Vec<Option<f64>> = vec
                .iter()
                .map(|v| {
                    if let GroupValue::F64(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Float64Array::from(arr)) as ArrayRef)
        }
        GroupValue::Str(_) => {
            let arr: Vec<Option<&str>> = vec
                .iter()
                .map(|v| {
                    if let GroupValue::Str(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::StringArray::from(arr)) as ArrayRef)
        }
        GroupValue::Bool(_) => {
            let arr: Vec<Option<bool>> = vec
                .iter()
                .map(|v| {
                    if let GroupValue::Bool(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::BooleanArray::from(arr)) as ArrayRef)
        }
        GroupValue::Null => {
            let len = vec.len();
            Ok(arrow::array::new_null_array(default_type, len))
        }
    }
}

fn collect_agg_column<'a, I>(agg: &Aggregation, it: I) -> Result<ArrayRef, String>
where
    I: Iterator<Item = &'a AggState>,
{
    let vec: Vec<&AggState> = it.collect();
    match agg.function {
        AggregateFunction::Count => {
            let arr: Vec<Option<i64>> = vec
                .iter()
                .map(|s| {
                    if let AggState::Count(c) = s {
                        Some(*c as i64)
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Int64Array::from(arr)) as ArrayRef)
        }
        AggregateFunction::Sum => {
            let arr: Vec<Option<f64>> = vec
                .iter()
                .map(|s| {
                    if let AggState::Sum(v) = s {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Float64Array::from(arr)) as ArrayRef)
        }
        AggregateFunction::Avg => {
            let arr: Vec<Option<f64>> = vec
                .iter()
                .map(|s| {
                    if let AggState::Avg { sum, count } = s {
                        if *count > 0 {
                            Some(sum / (*count as f64))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Float64Array::from(arr)) as ArrayRef)
        }
        AggregateFunction::Min => {
            let arr: Vec<Option<f64>> = vec
                .iter()
                .map(|s| {
                    if let AggState::Min(v) = s {
                        if v.is_finite() {
                            Some(*v)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Float64Array::from(arr)) as ArrayRef)
        }
        AggregateFunction::Max => {
            let arr: Vec<Option<f64>> = vec
                .iter()
                .map(|s| {
                    if let AggState::Max(v) = s {
                        if v.is_finite() {
                            Some(*v)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Arc::new(arrow::array::Float64Array::from(arr)) as ArrayRef)
        }
    }
}

impl Operator for AggregateOperator {
    fn execute(&self, input: &RecordBatch) -> Result<RecordBatch, String> {
        self.hash_aggregate(std::slice::from_ref(input))
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute_many(&self, inputs: &[RecordBatch]) -> Result<Vec<RecordBatch>, String> {
        let batch = self.hash_aggregate(inputs)?;
        Ok(if batch.is_empty() { vec![] } else { vec![batch] })
    }
}
