// Cross entropy forward over logits [batch, classes].
// Output is per-sample loss [batch]: logsumexp(logits[row]) - logits[row, target[row]]

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<i32>;
@group(0) @binding(2) var<storage, read_write> out_loss: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // [batch, classes, _, _]

@compute @workgroup_size(256)
fn cross_entropy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let batch = dims.x;
  let classes = dims.y;
  if (row >= batch) { return; }

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

  let t = u32(max(targets[row], 0));
  let target_idx = min(t, classes - 1u);
  let log_denom = maxv + log(sum_exp);
  out_loss[row] = log_denom - logits[base + target_idx];
}

