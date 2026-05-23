// Softmax shader for contiguous tensors with arbitrary rank and reduction dim.
// Flattened mapping:
// input shape: [d0, d1, ..., d{r-1}], reduce along axis with:
// outer = prod(d0..d{axis-1}), axis_size = d{axis}, inner = prod(d{axis+1}..d{r-1})
// rows = outer * inner
// row -> (outer_idx = row / inner, inner_idx = row % inner)
// linear index for class k in this row: outer_idx * axis_size * inner + k * inner + inner_idx

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // [rows, axis_size, inner, _]

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let rows = dims.x;
  let axis_size = dims.y;
  let inner = dims.z;
  if (row >= rows) { return; }
  if (axis_size == 0u || inner == 0u) { return; }

  let outer_idx = row / inner;
  let inner_idx = row % inner;
  let base = outer_idx * axis_size * inner + inner_idx;

  // Find max for numerical stability
  var maxv = input[base];
  var c: u32 = 1u;
  loop {
    if (c >= axis_size) { break; }
    let v = input[base + c * inner];
    if (v > maxv) { maxv = v; }
    c = c + 1u;
  }

  // Sum exp
  var sum_exp = 0.0;
  c = 0u;
  loop {
    if (c >= axis_size) { break; }
    sum_exp = sum_exp + exp(input[base + c * inner] - maxv);
    c = c + 1u;
  }

  // Normalize
  c = 0u;
  loop {
    if (c >= axis_size) { break; }
    output[base + c * inner] = exp(input[base + c * inner] - maxv) / sum_exp;
    c = c + 1u;
  }
}
