/**
 * Slice operation for tensors.
 * Supports multi-dimensional slicing with start:stop:step semantics.
 */

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Slice parameters for each dimension (up to 4D)
struct SliceParams {
  // Input shape (padded to 4D)
  input_shape: vec4<u32>,
  // Output shape (padded to 4D)
  output_shape: vec4<u32>,
  // Start indices for each dimension
  starts: vec4<i32>,
  // Step sizes for each dimension
  steps: vec4<i32>,
  // Number of dimensions
  ndim: u32,
  // Total output elements
  output_size: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: SliceParams;

@compute @workgroup_size(256)
fn slice(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.output_size) {
    return;
  }

  // Compute strides (left-aligned)
  var out_strides: array<u32, 4>;
  var in_strides: array<u32, 4>;
  
  for (var i = 0u; i < 4u; i++) {
    out_strides[i] = 1u;
    in_strides[i] = 1u;
  }

  if (params.ndim > 0u) {
    out_strides[params.ndim - 1u] = 1u;
    in_strides[params.ndim - 1u] = 1u;
    for (var d = i32(params.ndim) - 2; d >= 0; d--) {
      out_strides[d] = out_strides[d + 1] * params.output_shape[d + 1];
      in_strides[d] = in_strides[d + 1] * params.input_shape[d + 1];
    }
  }

  // Convert flat output index to multi-dimensional output coordinates
  var out_coords: array<u32, 4>;
  var remaining = idx;

  for (var d = 0u; d < params.ndim; d++) {
    out_coords[d] = remaining / out_strides[d];
    remaining = remaining % out_strides[d];
  }

  // Map output coordinates to input coordinates using slice params
  var in_idx = 0u;
  for (var d = 0u; d < params.ndim; d++) {
    let in_coord = u32(params.starts[d]) + out_coords[d] * u32(params.steps[d]);
    in_idx += in_coord * in_strides[d];
  }

  output[idx] = input[in_idx];
}
