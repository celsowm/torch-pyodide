// Softmax backward gradient computation.
// grad_input = softmax * (grad_output - sum(grad_output * softmax) along last dim)

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> softmax: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct Dims {
  batch_size: u32,
  num_classes: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn softmax_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = dims.batch_size * dims.num_classes;
  if (idx >= total) { return; }

  let row = idx / dims.num_classes;
  let col = idx % dims.num_classes;

  var dot: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.num_classes; k = k + 1u) {
    let offset = row * dims.num_classes + k;
    dot = dot + grad_output[offset] * softmax[offset];
  }

  let out_idx = row * dims.num_classes + col;
  let s = softmax[out_idx];
  grad_input[out_idx] = s * (grad_output[out_idx] - dot);
}

