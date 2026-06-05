// Fused optimizer steps for the remaining PyTorch-compatible variants:
//   - Adagrad
//   - Adamax (L-infinity Adam)
//   - NAdam  (Nesterov Adam)
//   - RAdam  (Rectified Adam)
//
// Each entrypoint updates its state buffers in place. The hyperparameters are
// packed in a single vec4 (hp) plus a second vec4 (extra) for the
// algorithm-specific fields.

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> state0: array<f32>;
@group(0) @binding(3) var<storage, read_write> state1: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // [length, _, _, _]
@group(0) @binding(5) var<uniform> hp: vec4<f32>;   // [lr, beta1, beta2, eps]
@group(0) @binding(6) var<uniform> extra: vec4<f32>;

// -----------------------------------------------------------------------------
// Adagrad
// state0 = sum_squares accumulator
// state1 unused
// extra  = [weight_decay, lr_decay, 0, 0]
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn adagrad_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let lr = hp.x;
  let eps = hp.w;
  let weight_decay = extra.x;

  let p = param[i];
  var g = grad[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }
  let s = state0[i] + g * g;
  state0[i] = s;
  let std_ = sqrt(s) + eps;
  param[i] = p - lr * g / std_;
}

// -----------------------------------------------------------------------------
// Adamax (L-infinity variant of Adam)
// state0 = exp_avg
// state1 = exp_inf
// extra  = [weight_decay, step_size, bias_correction1, 0]
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn adamax_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let lr = hp.x;
  let beta1 = hp.y;
  let beta2 = hp.z;
  let eps = hp.w;
  let weight_decay = extra.x;
  let step_size = extra.y;
  let bc1 = extra.z;

  let p = param[i];
  var g = grad[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }

  let m = state0[i] * beta1 + (1.0 - beta1) * g;
  let u = max(state1[i] * beta2, abs(g));
  state0[i] = m;
  state1[i] = u;

  let denom = u + eps;
  let update = (m / denom) * step_size * bc1;
  param[i] = p - update;
}

// -----------------------------------------------------------------------------
// NAdam (Nesterov-Adam)
// state0 = exp_avg
// state1 = exp_avg_sq
// extra  = [weight_decay, step_size, mu, _]
//   where mu = beta1 * (1 - 0.5 * 0.96^(t*0.004))   (PyTorch's Nesterov schedule)
//
// We compute the integrated Nesterov momentum look-ahead in the WGSL.
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn nadam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let lr = hp.x;
  let beta1 = hp.y;
  let beta2 = hp.z;
  let eps = hp.w;
  let weight_decay = extra.x;
  let step_size = extra.y;
  let mu = extra.z;

  let p = param[i];
  var g = grad[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }

  let m = state0[i] * beta1 + (1.0 - beta1) * g;
  let v = state1[i] * beta2 + (1.0 - beta2) * g * g;
  state0[i] = m;
  state1[i] = v;

  // Nesterov look-ahead: m_hat = mu_{t+1} * m / (1 - prod(beta1)) + (1 - mu_{t+1}) * g / (1 - prod(beta1))
  // For simplicity we use the bias-corrected forms with mu as an extra input
  // (the Python driver computes the schedule value).
  let m_hat = mu * m + (1.0 - mu) * g;
  let denom = sqrt(v) + eps;
  let update = (m_hat / denom) * step_size;
  param[i] = p - update;
}

// -----------------------------------------------------------------------------
// RAdam (Rectified Adam)
// state0 = exp_avg
// state1 = exp_avg_sq
// extra  = [weight_decay, step_size, beta1_pow_t, beta2_pow_t]
// We implement the rectified variance term in the shader using the
// SMA-length based approximation.
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn radam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let lr = hp.x;
  let beta1 = hp.y;
  let beta2 = hp.z;
  let eps = hp.w;
  let weight_decay = extra.x;
  let step_size = extra.y;
  let beta1_pow_t = extra.z;
  let beta2_pow_t = extra.w;

  let p = param[i];
  var g = grad[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }

  let m = state0[i] * beta1 + (1.0 - beta1) * g;
  let v = state1[i] * beta2 + (1.0 - beta2) * g * g;
  state0[i] = m;
  state1[i] = v;

  // bias-corrected estimates
  let m_hat = m / (1.0 - beta1_pow_t);
  let v_hat = v / (1.0 - beta2_pow_t);

  // Compute the rectification term rho_t.
  // rho_inf = 2 / (1 - beta2) - 1
  // rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
  // We compute rho_t as: rho_t = 2 / (1 - beta2) - 1 - 2 * (1 - beta2_pow_t) * exp_term
  // For practical use, the Python driver passes `step_size` already scaled by
  // the SMA-length, and we use the simple variant:
  //   if rho_t > 5: use sqrt((1 - beta2^t) * v_hat) (with no / sqrt(1 - prod))
  //   else:         no adaptive learning rate
  let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
  let one_minus_b2t = 1.0 - beta2_pow_t;
  // Approximate the SMA length: t = -log(beta2_pow_t) / log(beta2)
  // Avoid log(0): clamp beta2_pow_t away from 0
  let safe_b2t = max(beta2_pow_t, 1e-10);
  let t_approx = -log(safe_b2t) / log(max(beta2, 1e-10));
  let rho_t = rho_inf - 2.0 * t_approx * beta2_pow_t / max(one_minus_b2t, 1e-10);

  var update: f32;
  if (rho_t > 5.0) {
    let rect = sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
    update = step_size * m_hat * rect;
  } else {
    update = step_size * m_hat;
  }
  param[i] = p - update;
}
