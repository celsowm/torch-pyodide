@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;

struct Params {
  total_len: u32,
  output_capacity: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total_len) { return; }

  if (input[idx] != 0.0) {
    let pos = atomicAdd(&counter, 1u);
    // pos+1 to skip position 0 (which stores the total count).
    let write_pos = pos + 1u;
    if (write_pos < params.output_capacity) {
      output[write_pos] = f32(idx);
    }
  }
}
