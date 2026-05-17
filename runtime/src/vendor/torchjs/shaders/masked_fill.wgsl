struct Params {
  value: f32,
  size: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }

  // mask == 0 means we use original value, mask != 0 means we use fill value
  if (mask[idx] != 0.0) {
    output[idx] = params.value;
  } else {
    output[idx] = input[idx];
  }
}
