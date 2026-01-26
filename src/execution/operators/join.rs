// Hash joins (inner and left)

use crate::execution::batch::{RecordBatch, SchemaRef};
use crate::planner::logical_plan::JoinType;
use arrow::array::ArrayRef;
use arrow::datatypes::DataType;
use std::collections::HashMap;
use std::sync::Arc;

/// Hash join: build a hash table from the right (build) side, probe with the left.
/// Supports Inner and Left join.
pub struct HashJoinOperator {
    left_key: String,
    right_key: String,
    join_type: JoinType,
    /// Output schema: left fields + right fields
    schema: SchemaRef,
}

impl HashJoinOperator {
    /// Create a new HashJoin operator. left_schema and right_schema are used to build output schema.
    pub fn new(
        left_key: String,
        right_key: String,
        join_type: JoinType,
        left_schema: SchemaRef,
        right_schema: SchemaRef,
    ) -> Result<Self, String> {
        let mut fields = left_schema.fields().iter().map(|f| f.as_ref().clone()).collect::<Vec<_>>();
        fields.extend(right_schema.fields().iter().map(|f| f.as_ref().clone()));
        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        Ok(Self {
            left_key,
            right_key,
            join_type,
            schema,
        })
    }

    /// Execute the join. Both sides are concat'd to single batches, then hash join.
    pub fn execute_join(
        &self,
        left_batches: &[RecordBatch],
        right_batches: &[RecordBatch],
    ) -> Result<Vec<RecordBatch>, String> {
        let left = if left_batches.is_empty() {
            return Ok(Vec::new());
        } else if left_batches.len() == 1 {
            left_batches[0].clone()
        } else {
            RecordBatch::concat(left_batches)?
        };

        let right = if right_batches.is_empty() {
            if matches!(self.join_type, JoinType::Left) {
                // Left join with empty right: return left with nulls for right cols
                return self.left_only_result(&left);
            }
            return Ok(Vec::new());
        } else if right_batches.len() == 1 {
            right_batches[0].clone()
        } else {
            RecordBatch::concat(right_batches)?
        };

        // Build: hash map from right key -> right row indices
        let right_col = right
            .column_by_name(&self.right_key)
            .ok_or_else(|| format!("Right key '{}' not found", self.right_key))?;
        let mut map: HashMap<String, Vec<usize>> = HashMap::new();
        for row in 0..right.num_rows() {
            let k = key_string(right_col, row)?;
            map.entry(k).or_default().push(row);
        }

        // Probe: for each left row, find matches
        let left_col = left
            .column_by_name(&self.left_key)
            .ok_or_else(|| format!("Left key '{}' not found", self.left_key))?;

        let mut left_indices = Vec::new();
        let mut right_indices: Vec<Option<usize>> = Vec::new();

        for lr in 0..left.num_rows() {
            let k = key_string(left_col, lr)?;
            if let Some(rows) = map.get(&k) {
                for &rr in rows {
                    left_indices.push(lr as u32);
                    right_indices.push(Some(rr));
                }
            } else if matches!(self.join_type, JoinType::Left) {
                left_indices.push(lr as u32);
                right_indices.push(None);
            }
        }

        if left_indices.is_empty() {
            return Ok(vec![]);
        }

        // Build output: take left columns by left_indices; for right, take or null
        let u32_indices = arrow::array::UInt32Array::from(left_indices.clone());
        let left_cols: Vec<ArrayRef> = left
            .columns()
            .iter()
            .map(|c| arrow_select::take::take(c.as_ref(), &u32_indices, None).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        let num_left = left.schema().fields().len();
        let right_cols: Vec<ArrayRef> = right
            .columns()
            .iter()
            .enumerate()
            .map(|(i, c)| {
                build_with_nulls(c.as_ref(), &right_indices).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut all_cols = left_cols;
        all_cols.extend(right_cols);
        let out = RecordBatch::try_new(self.schema.clone(), all_cols)?;
        Ok(vec![out])
    }

    /// Left join with empty right: left with nulls for right columns (from output schema)
    fn left_only_result(&self, left: &RecordBatch) -> Result<Vec<RecordBatch>, String> {
        let num_left = left.schema().fields().len();
        let mut cols = left.columns().to_vec();
        for i in num_left..self.schema.fields().len() {
            let f = self.schema.fields()[i].as_ref();
            cols.push(arrow::array::new_null_array(f.data_type(), left.num_rows()));
        }
        let batch = RecordBatch::try_new(self.schema.clone(), cols)?;
        Ok(vec![batch])
    }
}

fn key_string(col: &ArrayRef, row: usize) -> Result<String, String> {
    use arrow::array::*;
    if col.is_null(row) {
        return Ok("__NULL__".to_string());
    }
    match col.data_type() {
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().ok_or("Int32")?;
            Ok(format!("i32:{}", a.value(row)))
        }
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().ok_or("Int64")?;
            Ok(format!("i64:{}", a.value(row)))
        }
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().ok_or("Float64")?;
            Ok(format!("f64:{}", a.value(row)))
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<StringArray>().ok_or("Utf8")?;
            Ok(format!("str:{}", a.value(row)))
        }
        DataType::Boolean => {
            let a = col.as_any().downcast_ref::<BooleanArray>().ok_or("Bool")?;
            Ok(format!("bool:{}", a.value(row)))
        }
        _ => Err(format!("Unsupported join key type: {:?}", col.data_type())),
    }
}

/// Build array from `base` by indexing with `indices`; None means null in output.
fn build_with_nulls(base: &dyn arrow::array::Array, indices: &[Option<usize>]) -> Result<ArrayRef, String> {
    use arrow::array::*;
    match base.data_type() {
        DataType::Int32 => {
            let a = base.as_any().downcast_ref::<Int32Array>().ok_or("Int32")?;
            let out: Vec<Option<i32>> = indices.iter().map(|o| o.and_then(|i| if a.is_null(i) { None } else { Some(a.value(i)) })).collect();
            Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
        }
        DataType::Int64 => {
            let a = base.as_any().downcast_ref::<Int64Array>().ok_or("Int64")?;
            let out: Vec<Option<i64>> = indices.iter().map(|o| o.and_then(|i| if a.is_null(i) { None } else { Some(a.value(i)) })).collect();
            Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
        }
        DataType::Float64 => {
            let a = base.as_any().downcast_ref::<Float64Array>().ok_or("Float64")?;
            let out: Vec<Option<f64>> = indices.iter().map(|o| o.and_then(|i| if a.is_null(i) { None } else { Some(a.value(i)) })).collect();
            Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let a = base.as_any().downcast_ref::<StringArray>().ok_or("Utf8")?;
            let out: Vec<Option<&str>> = indices.iter().map(|o| o.and_then(|i| if a.is_null(i) { None } else { Some(a.value(i)) })).collect();
            Ok(Arc::new(StringArray::from(out)) as ArrayRef)
        }
        DataType::Boolean => {
            let a = base.as_any().downcast_ref::<BooleanArray>().ok_or("Bool")?;
            let out: Vec<Option<bool>> = indices.iter().map(|o| o.and_then(|i| if a.is_null(i) { None } else { Some(a.value(i)) })).collect();
            Ok(Arc::new(BooleanArray::from(out)) as ArrayRef)
        }
        _ => Err(format!("Unsupported type in build_with_nulls: {:?}", base.data_type())),
    }
}
