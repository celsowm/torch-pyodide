@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Flat roll: for each input element at position idx,
// writes it to output[(idx + shift) % total_len].
// shift is passed as the first uniform value.

struct Params {
  shift: i32,
  total_len: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total_len) { return; }

  // Compute destination index with positive modulo.
  let dst = (i32(idx) + params.shift) % i32(params.total_len);
  let dst_u = select(u32(dst + i32(params.total_len)), u32(dst), dst >= 0);

  output[dst_u] = input[idx];
}
