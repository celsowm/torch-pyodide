@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
  batch_size: u32,
  normalized_size: u32,
  eps: f32,
  _pad: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch_idx = gid.x;
  if (batch_idx >= params.batch_size) { return; }

  let offset = batch_idx * params.normalized_size;

  // Compute mean
  var sum = 0.0;
  for (var i = 0u; i < params.normalized_size; i++) {
    sum += input[offset + i];
  }
  let mean = sum / f32(params.normalized_size);

  // Compute variance
  var var_sum = 0.0;
  for (var i = 0u; i < params.normalized_size; i++) {
    let diff = input[offset + i] - mean;
    var_sum += diff * diff;
  }
  let variance = var_sum / f32(params.normalized_size);
  let inv_std = 1.0 / sqrt(variance + params.eps);

  // Normalize and apply affine transform
  for (var i = 0u; i < params.normalized_size; i++) {
    let normalized = (input[offset + i] - mean) * inv_std;
    output[offset + i] = gamma[i] * normalized + beta[i];
  }
}
