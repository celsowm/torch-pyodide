/**
 * Index select (gather) operation along a dimension.
 * For 2D tensors: output[i, j] = input[indices[i], j] when dim=0
 *                 output[i, j] = input[i, indices[j]] when dim=1
 */

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  dim: u32,           // Dimension to index along
  input_dim0: u32,    // Input shape[0]
  input_dim1: u32,    // Input shape[1]
  num_indices: u32,   // Number of indices
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn index_select_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_size = select(params.num_indices * params.input_dim1,
                           params.input_dim0 * params.num_indices,
                           params.dim == 0u);
  let idx = global_id.x;
  if (idx >= output_size) {
    return;
  }

  if (params.dim == 0u) {
    // Indexing along dim 0: output[i, j] = input[indices[i], j]
    let i = idx / params.input_dim1;
    let j = idx % params.input_dim1;
    let input_row = u32(indices[i]);
    output[idx] = input[input_row * params.input_dim1 + j];
  } else {
    // Indexing along dim 1: output[i, j] = input[i, indices[j]]
    let i = idx / params.num_indices;
    let j = idx % params.num_indices;
    let input_col = u32(indices[j]);
    output[idx] = input[i * params.input_dim1 + input_col];
  }
}

// For 1D tensors: simple gather
@compute @workgroup_size(256)
fn index_select_1d(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_indices) {
    return;
  }

  let input_idx = u32(indices[idx]);
  output[idx] = input[input_idx];
}
