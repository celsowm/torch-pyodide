// Cross entropy backward over logits [batch, classes].
// grad_input[row, col] = (softmax(logits[row])[col] - one_hot(target[row], col)) * row_scale
// row_scale depends on reduction and grad_output.

@group(0) @binding(0) var<storage, read> grad_output: array<f32>; // scalar (len=1) or vector [batch]
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>;   // [batch, classes, reduction_mode, grad_is_scalar]
@group(0) @binding(5) var<uniform> scales: vec4<f32>; // [norm_scale, _, _, _]

@compute @workgroup_size(256)
fn cross_entropy_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let batch = dims.x;
  let classes = dims.y;
  let total = batch * classes;
  if (idx >= total) { return; }

  let row = idx / classes;
  let col = idx % classes;
  let base = row * classes;

  var maxv = logits[base];
  var c: u32 = 1u;
  loop {
    if (c >= classes) { break; }
    let v = logits[base + c];
    if (v > maxv) { maxv = v; }
    c = c + 1u;
  }

  var sum_exp = 0.0;
  c = 0u;
  loop {
    if (c >= classes) { break; }
    sum_exp = sum_exp + exp(logits[base + c] - maxv);
    c = c + 1u;
  }

  let prob = exp(logits[idx] - maxv) / sum_exp;
  let t = i32(targets[row]);
  let target_idx = u32(max(t, 0));
  let final_target_idx = min(target_idx, classes - 1u);
  let one_hot = select(0.0, 1.0, col == final_target_idx);

  var upstream = 0.0;
  if (dims.w == 1u) {
    upstream = grad_output[0];
  } else {
    upstream = grad_output[row];
  }

  let row_scale = upstream * scales.x;
  grad_input[idx] = (prob - one_hot) * row_scale;
}

