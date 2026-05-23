// Backward for global max/min reduction.
// mode: 0=max, 1=min

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>; // scalar
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // [length, mode, _, _]

@compute @workgroup_size(1)
fn maxmin_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  let n = params.x;
  let mode = params.y;
  if (n == 0u) { return; }

  var best_idx: u32 = 0u;
  var best_val: f32 = input[0];
  var i: u32 = 1u;
  loop {
    if (i >= n) { break; }
    let v = input[i];
    if ((mode == 0u && v > best_val) || (mode == 1u && v < best_val)) {
      best_val = v;
      best_idx = i;
    }
    i = i + 1u;
  }

  i = 0u;
  loop {
    if (i >= n) { break; }
    grad_input[i] = 0.0;
    i = i + 1u;
  }
  grad_input[best_idx] = grad_output[0];
}

