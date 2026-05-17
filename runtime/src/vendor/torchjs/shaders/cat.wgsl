struct CatParams {
  a_shape: vec4<u32>,
  b_shape: vec4<u32>,
  out_shape: vec4<u32>,
  dim: u32,
  ndim: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: CatParams;

fn get_stride(shape: vec4<u32>, d: u32) -> u32 {
  var s = 1u;
  for (var i = 3u; i > d; i--) {
    s = s * shape[i];
  }
  return s;
}

fn flat_index(coords: vec4<u32>, shape: vec4<u32>) -> u32 {
  var idx = 0u;
  var stride = 1u;
  for (var i = 3u; ; i--) {
    idx = idx + coords[i] * stride;
    if (i == 0u) { break; }
    stride = stride * shape[i];
  }
  return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.out_shape[0] * params.out_shape[1] * params.out_shape[2] * params.out_shape[3]) {
    return;
  }

  var remaining = idx;
  var coords: vec4<u32>;
  for (var i = 0u; i < 4u; i++) {
    let stride = get_stride(params.out_shape, i);
    coords[i] = remaining / stride;
    remaining = remaining % stride;
  }

  let dim_coord = coords[params.dim];
  let dim_size_a = params.a_shape[params.dim];

  if (dim_coord < dim_size_a) {
    let a_idx = flat_index(coords, params.a_shape);
    output[idx] = input_a[a_idx];
  } else {
    var b_coords = coords;
    b_coords[params.dim] = dim_coord - dim_size_a;
    let b_idx = flat_index(b_coords, params.b_shape);
    output[idx] = input_b[b_idx];
  }
}
