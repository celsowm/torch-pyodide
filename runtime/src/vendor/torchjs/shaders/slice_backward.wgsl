/**
 * Slice backward operation for tensors.
 * Maps gradients from sliced shape back to original shape.
 */

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;

// Slice parameters for each dimension (up to 4D)
struct SliceParams {
  // Input shape (the sliced tensor shape, padded to 4D)
  input_shape: vec4<u32>,
  // Output shape (the original full tensor shape, padded to 4D)
  output_shape: vec4<u32>,
  // Start indices for each dimension
  starts: vec4<i32>,
  // Step sizes for each dimension
  steps: vec4<i32>,
  // Number of dimensions
  ndim: u32,
  // Total elements in the grad_output (sliced tensor)
  grad_size: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: SliceParams;

@compute @workgroup_size(256)
fn slice_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.grad_size) {
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
      // output_shape is the original tensor shape
      out_strides[d] = out_strides[d + 1] * params.output_shape[d + 1];
      // input_shape is the sliced tensor shape
      in_strides[d] = in_strides[d + 1] * params.input_shape[d + 1];
    }
  }

  // Convert flat index of grad_output (which is sliced shape) to multi-dimensional coordinates
  var in_coords: array<u32, 4>;
  var remaining = idx;

  for (var d = 0u; d < params.ndim; d++) {
    in_coords[d] = remaining / in_strides[d];
    remaining = remaining % in_strides[d];
  }

  // Map coordinates to original tensor coordinates
  var out_idx = 0u;
  for (var d = 0u; d < params.ndim; d++) {
    let out_coord = u32(params.starts[d]) + in_coords[d] * u32(params.steps[d]);
    out_idx += out_coord * out_strides[d];
  }

  // Atomically add the gradient (since multiple sliced indices could theoretically map to same original index if step=0, but step > 0 normally)
  // For standard slice, it's safe to just assign, but we can just use simple assignment since slices are unique.
  grad_input[out_idx] = grad_output[idx];
}
