struct PermuteParams {
  out_shape: vec4<u32>,
  src_strides: vec4<u32>,
  out_strides: vec4<u32>,
  ndim: u32,
  total: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: PermuteParams;

// Permutation: dims[0..ndim-1] stored in a buffer
@group(0) @binding(1) var<storage, read> perm: array<u32>;

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

  // Convert dst_idx to output coordinates
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

  var src_strides_arr: array<u32, 4>;
  src_strides_arr[0] = params.src_strides[0];
  src_strides_arr[1] = params.src_strides[1];
  src_strides_arr[2] = params.src_strides[2];
  src_strides_arr[3] = params.src_strides[3];

  var out_coords: array<u32, 4>;
  var rem = dst_idx;
  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    out_coords[dim_idx] = rem / out_strides_arr[dim_idx];
    rem = rem % out_strides_arr[dim_idx];
  }

  // Apply inverse permutation to get source coords
  // perm[i] tells which output dim maps to source dim i
  // So source coord[d] = output coords[inv_perm[d]]
  // We compute inv_perm: inv_perm[perm[i]] = i
  var inv_perm: array<u32, 4>;
  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    inv_perm[perm[d]] = dim_idx;
  }

  var src_coords: array<u32, 4>;
  for (var d = 0u; d < params.ndim; d++) {
    let dim_idx = offset + d;
    src_coords[dim_idx] = out_coords[inv_perm[dim_idx]];
  }

  let src_idx = coord_to_flat(src_coords, src_strides_arr);
  output[dst_idx] = input[src_idx];
}
