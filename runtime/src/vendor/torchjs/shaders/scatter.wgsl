@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;

// Flat scatter: for each element i at linear position idx,
// write src[idx] to output[uint(indices[idx])].
// Output must already contain a copy of the input (done via buffer copy before dispatch).

struct Params {
  index_len: u32,
  output_len: u32,
  src_len: u32,
  _pad: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.index_len) { return; }
  if (idx >= params.src_len) { return; }

  let pos = u32(indices[idx]);
  if (pos < params.output_len) {
    output[pos] = src[idx];
  }
}
