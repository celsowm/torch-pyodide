/**
 * Element-wise binary operations with broadcasting support.
 * Supports up to 4D tensors with NumPy-style broadcasting.
 */

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct BroadcastParams {
  // Output shape (padded to 4D, right-aligned)
  output_shape: vec4<u32>,
  // Input A strides (0 means broadcast this dim)
  a_strides: vec4<u32>,
  // Input B strides (0 means broadcast this dim)
  b_strides: vec4<u32>,
  // Number of dimensions
  ndim: u32,
  // Total output elements
  output_size: u32,
  // Operation type: 0=add, 1=sub, 2=mul, 3=div
  op: u32,
  _pad: u32,
}

@group(0) @binding(3) var<uniform> params: BroadcastParams;

@compute @workgroup_size(256)
fn broadcast_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.output_size) {
    return;
  }

  // Convert flat index to multi-dimensional coordinates
  let offset = 4u - params.ndim;
  var remaining = idx;
  var a_idx = 0u;
  var b_idx = 0u;

  // Compute output strides
  var out_strides: vec4<u32>;
  out_strides[3] = 1u;
  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }
  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }
  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }

  // For each dimension, compute coordinate and add to input indices
  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    let coord = remaining / out_strides[dim_idx];
    remaining = remaining % out_strides[dim_idx];

    a_idx += coord * params.a_strides[dim_idx];
    b_idx += coord * params.b_strides[dim_idx];
  }

  output[idx] = a[a_idx] + b[b_idx];
}

@compute @workgroup_size(256)
fn broadcast_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.output_size) {
    return;
  }

  let offset = 4u - params.ndim;
  var remaining = idx;
  var a_idx = 0u;
  var b_idx = 0u;

  var out_strides: vec4<u32>;
  out_strides[3] = 1u;
  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }
  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }
  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }

  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    let coord = remaining / out_strides[dim_idx];
    remaining = remaining % out_strides[dim_idx];
    a_idx += coord * params.a_strides[dim_idx];
    b_idx += coord * params.b_strides[dim_idx];
  }

  output[idx] = a[a_idx] - b[b_idx];
}

@compute @workgroup_size(256)
fn broadcast_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.output_size) {
    return;
  }

  let offset = 4u - params.ndim;
  var remaining = idx;
  var a_idx = 0u;
  var b_idx = 0u;

  var out_strides: vec4<u32>;
  out_strides[3] = 1u;
  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }
  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }
  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }

  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    let coord = remaining / out_strides[dim_idx];
    remaining = remaining % out_strides[dim_idx];
    a_idx += coord * params.a_strides[dim_idx];
    b_idx += coord * params.b_strides[dim_idx];
  }

  output[idx] = a[a_idx] * b[b_idx];
}

@compute @workgroup_size(256)
fn broadcast_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.output_size) {
    return;
  }

  let offset = 4u - params.ndim;
  var remaining = idx;
  var a_idx = 0u;
  var b_idx = 0u;

  var out_strides: vec4<u32>;
  out_strides[3] = 1u;
  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }
  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }
  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }

  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    let coord = remaining / out_strides[dim_idx];
    remaining = remaining % out_strides[dim_idx];
    a_idx += coord * params.a_strides[dim_idx];
    b_idx += coord * params.b_strides[dim_idx];
  }

  output[idx] = a[a_idx] / b[b_idx];
}

@compute @workgroup_size(256)
fn broadcast_pow(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.output_size) {
    return;
  }

  let offset = 4u - params.ndim;
  var remaining = idx;
  var a_idx = 0u;
  var b_idx = 0u;

  var out_strides: vec4<u32>;
  out_strides[3] = 1u;
  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }
  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }
  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }

  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    let coord = remaining / out_strides[dim_idx];
    remaining = remaining % out_strides[dim_idx];
    a_idx += coord * params.a_strides[dim_idx];
    b_idx += coord * params.b_strides[dim_idx];
  }

  output[idx] = pow(a[a_idx], b[b_idx]);
}
