// NLL Loss backward shader
// Computes gradient: -1/batch_size at target index, 0 elsewhere

@group(0) @binding(0) var<storage, read> targets: array<i32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;

struct Params {
  batch_size: u32,
  num_classes: u32,
  scale: f32,  // 1.0 for sum reduction, 1/batch_size for mean
  _pad: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn nll_loss_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_size = params.batch_size * params.num_classes;

  if (idx >= total_size) {
    return;
  }

  // Compute batch index and class index
  let batch_idx = idx / params.num_classes;
  let class_idx = idx % params.num_classes;

  // Get target for this batch
  let target_class = u32(targets[batch_idx]);

  // Gradient is -scale at target index, 0 elsewhere
  if (class_idx == target_class) {
    grad_input[idx] = -params.scale;
  } else {
    grad_input[idx] = 0.0;
  }
}
