struct RepeatParams {
  in_shape: vec4<u32>,
  out_shape: vec4<u32>,
  in_strides: vec4<u32>,
  out_strides: vec4<u32>,
  repeats: vec4<u32>,
  ndim: u32,
  total: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: RepeatParams;

fn coord_to_flat(coords: array<u32, 4>, strides: array<u32, 4>) -> u32 {
  var idx = 0u;
  for (var d = 0u; d < 4u; d++) {
    idx = idx + coords[d] * strides[d];
  }
  return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dst_idx = gid.x;
  if (dst_idx >= params.total) { return; }
  let offset = 4u - params.ndim;

  var out_strides_arr: array<u32, 4>;
  out_strides_arr[0] = params.out_strides[0];
  out_strides_arr[1] = params.out_strides[1];
  out_strides_arr[2] = params.out_strides[2];
  out_strides_arr[3] = params.out_strides[3];

  var out_shape_arr: array<u32, 4>;
  out_shape_arr[0] = params.out_shape[0];
  out_shape_arr[1] = params.out_shape[1];
  out_shape_arr[2] = params.out_shape[2];
  out_shape_arr[3] = params.out_shape[3];

  var in_strides_arr: array<u32, 4>;
  in_strides_arr[0] = params.in_strides[0];
  in_strides_arr[1] = params.in_strides[1];
  in_strides_arr[2] = params.in_strides[2];
  in_strides_arr[3] = params.in_strides[3];

  var in_shape_arr: array<u32, 4>;
  in_shape_arr[0] = params.in_shape[0];
  in_shape_arr[1] = params.in_shape[1];
  in_shape_arr[2] = params.in_shape[2];
  in_shape_arr[3] = params.in_shape[3];

  var repeats_arr: array<u32, 4>;
  repeats_arr[0] = params.repeats[0];
  repeats_arr[1] = params.repeats[1];
  repeats_arr[2] = params.repeats[2];
  repeats_arr[3] = params.repeats[3];

  // Convert dst_idx to output coordinates
  var out_coords: array<u32, 4>;
  var rem = dst_idx;
  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    out_coords[dim_idx] = rem / out_strides_arr[dim_idx];
    rem = rem % out_strides_arr[dim_idx];
  }

  // Map output coords to input coords via modulo (repeat)
  var in_coords: array<u32, 4>;
  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    in_coords[dim_idx] = out_coords[dim_idx] % in_shape_arr[dim_idx];
  }

  let src_idx = coord_to_flat(in_coords, in_strides_arr);
  output[dst_idx] = input[src_idx];
}
