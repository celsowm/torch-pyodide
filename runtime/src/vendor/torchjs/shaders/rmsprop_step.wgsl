// Fused RMSprop step with optional momentum.

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> square_avg: array<f32>;
@group(0) @binding(3) var<storage, read_write> momentum_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // [length, use_momentum, _, _]
@group(0) @binding(5) var<uniform> hp: vec4<f32>;   // [lr, alpha, eps, weight_decay]
@group(0) @binding(6) var<uniform> extra: vec4<f32>; // [momentum, _, _, _]

@compute @workgroup_size(256)
fn rmsprop_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let use_momentum = dims.y == 1u;
  let lr = hp.x;
  let alpha = hp.y;
  let eps = hp.z;
  let weight_decay = hp.w;
  let momentum = extra.x;

  let p = param[i];
  var g = grad[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }

  let sq = square_avg[i] * alpha + (1.0 - alpha) * g * g;
  square_avg[i] = sq;
  let denom = sqrt(sq) + eps;

  var update = g / denom;
  if (use_momentum) {
    let buf = momentum_buf[i] * momentum + update;
    momentum_buf[i] = buf;
    update = buf;
  }

  param[i] = p - lr * update;
}

