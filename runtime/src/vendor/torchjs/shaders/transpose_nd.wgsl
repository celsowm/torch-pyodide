struct Params {
  src_strides: vec4<u32>,
  dst_shape: vec4<u32>,
  dst_strides: vec4<u32>,
  ndim: u32,
  total: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dst_idx = gid.x;
  if (dst_idx >= params.total) { return; }

  // Convert dst_idx to coordinates
  var coords: array<u32, 4>;
  var rem = dst_idx;
  for (var d = i32(params.ndim) - 1; d >= 0; d--) {
    coords[d] = rem % params.dst_shape[d];
    rem = rem / params.dst_shape[d];
  }

  // Compute src index using transposed strides
  var src_idx = 0u;
  for (var d = 0u; d < params.ndim; d++) {
    src_idx += coords[d] * params.src_strides[d];
  }

  output[dst_idx] = input[src_idx];
}
