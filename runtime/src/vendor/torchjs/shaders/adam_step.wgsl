// Fused Adam step: updates parameter, exp_avg, exp_avg_sq in one pass.

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> exp_avg: array<f32>;
@group(0) @binding(3) var<storage, read_write> exp_avg_sq: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // [length, _, _, _]
@group(0) @binding(5) var<uniform> hp: vec4<f32>;   // [lr, beta1, beta2, eps]
@group(0) @binding(6) var<uniform> extra: vec4<f32>; // [weight_decay, step_size, inv_sqrt_bc2, _]

@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let beta1 = hp.y;
  let beta2 = hp.z;
  let eps = hp.w;
  let weight_decay = extra.x;
  let step_size = extra.y;
  let inv_sqrt_bc2 = extra.z;

  let p = param[i];
  var g = grad[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }

  let m = exp_avg[i] * beta1 + (1.0 - beta1) * g;
  let v = exp_avg_sq[i] * beta2 + (1.0 - beta2) * g * g;

  exp_avg[i] = m;
  exp_avg_sq[i] = v;

  let denom = sqrt(v) * inv_sqrt_bc2 + eps;
  let update = (m / denom) * step_size;
  param[i] = p - update;
}
