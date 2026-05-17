@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  diagonal: i32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.rows * params.cols;
  if (idx >= total) { return; }

  let row = idx / params.cols;
  let col = idx % params.cols;

  // Keep element if col <= row + diagonal
  if (i32(col) <= i32(row) + params.diagonal) {
    output[idx] = input[idx];
  } else {
    output[idx] = 0.0;
  }
}
