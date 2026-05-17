struct SelectParams {
  in_shape: vec4<u32>,
  out_shape: vec4<u32>,
  dim: u32,
  index: u32,
  ndim: u32,
  total: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SelectParams;

fn flat_index(coords: ptr<function, array<u32, 4>>, shape: ptr<function, array<u32, 4>>, ndim: u32) -> u32 {
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
  if (out_flat >= params.total) { return; }

  // Convert output flat index to output coordinates
  var out_shape_arr: array<u32, 4>;
  out_shape_arr[0] = params.out_shape[0];
  out_shape_arr[1] = params.out_shape[1];
  out_shape_arr[2] = params.out_shape[2];
  out_shape_arr[3] = params.out_shape[3];

  var in_shape_arr: array<u32, 4>;
  in_shape_arr[0] = params.in_shape[0];
  in_shape_arr[1] = params.in_shape[1];
  in_shape_arr[2] = params.in_shape[2];
  in_shape_arr[3] = params.in_shape[3];

  var out_coords: array<u32, 4>;
  var rem = out_flat;
  var stride = params.total;
  for (var d = 0u; d < params.ndim - 1u; d++) {
    stride = stride / out_shape_arr[d];
    out_coords[d] = rem / stride;
    rem = rem % stride;
  }

  // Build input coordinates by inserting the index at the select dim
  var in_coords: array<u32, 4>;
  for (var d = 0u; d < params.dim; d++) {
    in_coords[d] = out_coords[d];
  }
  in_coords[params.dim] = params.index;
  for (var d = params.dim; d < params.ndim - 1u; d++) {
    in_coords[d + 1] = out_coords[d];
  }

  let in_idx = flat_index(&in_coords, &in_shape_arr, params.ndim);
  output[out_flat] = input[in_idx];
}
