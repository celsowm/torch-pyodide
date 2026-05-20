@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Input shapes/strides use left-padding (padShapeTo4Left): shape[3,2] -> [3,2,1,1]
// The shader iterates rows->cols->depth->..., so shape[0] is most significant dim.
// For indices/output shape, same left-padding convention.

struct Params {
  dim: u32,
  rank: u32,
  output_len: u32,
  _pad: u32,
  in_shape0: u32,
  in_stride0: u32,
  in_shape1: u32,
  in_stride1: u32,
  in_shape2: u32,
  in_stride2: u32,
  in_shape3: u32,
  in_stride3: u32,
  out_shape0: u32,
  out_shape1: u32,
  out_shape2: u32,
  out_shape3: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

fn inStride(i: u32) -> u32 {
  if (i == 0u) { return params.in_stride0; }
  if (i == 1u) { return params.in_stride1; }
  if (i == 2u) { return params.in_stride2; }
  return params.in_stride3;
}

fn outShape(i: u32) -> u32 {
  if (i == 0u) { return params.out_shape0; }
  if (i == 1u) { return params.out_shape1; }
  if (i == 2u) { return params.out_shape2; }
  return params.out_shape3;
}

fn inShape(i: u32) -> u32 {
  if (i == 0u) { return params.in_shape0; }
  if (i == 1u) { return params.in_shape1; }
  if (i == 2u) { return params.in_shape2; }
  return params.in_shape3;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.output_len) { return; }

  var remaining = idx;
  var src_linear: u32 = 0u;

  // Iterate from last dim (innermost/col) to first dim (outermost/row)
  for (var d = 0u; d < params.rank; d += 1u) {
    let dim_idx = params.rank - 1u - d;
    let sz = outShape(dim_idx);
    let coord = remaining % sz;
    remaining = remaining / sz;

    if (dim_idx == params.dim) {
      let gather_idx = u32(indices[idx]);
      src_linear = src_linear + gather_idx * inStride(dim_idx);
    } else {
      src_linear = src_linear + coord * inStride(dim_idx);
    }
  }

  output[idx] = input[src_linear];
}
