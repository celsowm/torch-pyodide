/**
 * Log softmax backward gradient computation.
 * grad[i,j] = gradOutput[i,j] - softmax[i,j] * sum_k(gradOutput[i,k])
 * @status implemented
 */

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> softmax: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct Dims {
  batch_size: u32,
  num_classes: u32,
}
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn log_softmax_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = dims.batch_size * dims.num_classes;
  if (idx >= total) { return; }

  let i = idx / dims.num_classes;  // batch index
  let j = idx % dims.num_classes;  // class index

  // Sum gradOutput along row i
  var grad_sum = 0.0;
  for (var k = 0u; k < dims.num_classes; k++) {
    grad_sum += grad_output[i * dims.num_classes + k];
  }

  // grad[i,j] = gradOutput[i,j] - softmax[i,j] * grad_sum
  grad_input[idx] = grad_output[idx] - softmax[idx] * grad_sum;
}
