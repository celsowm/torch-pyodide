struct StackParams {
  in_shape: vec4<u32>,
  out_shape: vec4<u32>,
  dim: u32,
  ndim: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: StackParams;

fn flat_index(coords: ptr<function, array<u32, 5>>, shape: ptr<function, array<u32, 5>>, ndim: u32) -> u32 {
  var idx = 0u;
  var stride = 1u;
  var i = ndim;
  loop {
    if (i == 0u) { break; }
    i = i - 1u;
    idx = idx + (*coords)[i] * stride;
    stride = stride * (*shape)[i];
  }
  return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_flat = gid.x;
  var out_shape: array<u32, 5>;
  out_shape[0] = params.out_shape[0];
  out_shape[1] = params.out_shape[1];
  out_shape[2] = params.out_shape[2];
  out_shape[3] = params.out_shape[3];
  out_shape[4] = 1u;

  var in_shape: array<u32, 5>;
  in_shape[0] = params.in_shape[0];
  in_shape[1] = params.in_shape[1];
  in_shape[2] = params.in_shape[2];
  in_shape[3] = params.in_shape[3];
  in_shape[4] = 1u;

  // Total output elements
  var total = 1u;
  for (var i = 0u; i < params.ndim + 1u; i++) {
    total = total * out_shape[i];
  }
  if (out_flat >= total) { return; }

  // Convert flat index to output coords
  var out_coords: array<u32, 5>;
  var rem = out_flat;
  var stride = total;
  for (var i = 0u; i < params.ndim + 1u; i++) {
    stride = stride / out_shape[i];
    out_coords[i] = rem / stride;
    rem = rem % stride;
  }

  // which input tensor?
  let which = out_coords[params.dim];

  // Build input coords by removing the stack dim
  var in_coords: array<u32, 5>;
  for (var d = 0u; d < params.dim; d++) {
    in_coords[d] = out_coords[d];
  }
  for (var d = params.dim; d < params.ndim; d++) {
    in_coords[d] = out_coords[d + 1];
  }
  in_coords[params.ndim] = 0u;

  let in_idx = flat_index(&in_coords, &in_shape, params.ndim);

  if (which == 0u) {
    output[out_flat] = input_a[in_idx];
  } else {
    output[out_flat] = input_b[in_idx];
  }
}
