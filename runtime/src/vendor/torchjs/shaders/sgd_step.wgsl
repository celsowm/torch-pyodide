// Fused SGD step with optional momentum and nesterov.

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> momentum_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // [length, has_momentum, nesterov, _]
@group(0) @binding(4) var<uniform> hp: vec4<f32>;   // [lr, momentum, weight_decay, dampening]

@compute @workgroup_size(256)
fn sgd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.x;
  if (i >= n) { return; }

  let has_momentum = dims.y == 1u;
  let nesterov = dims.z == 1u;

  let lr = hp.x;
  let momentum = hp.y;
  let weight_decay = hp.z;
  let dampening = hp.w;

  var g = grad[i];
  let p = param[i];
  if (weight_decay != 0.0) {
    g = g + weight_decay * p;
  }

  var update = g;
  if (has_momentum) {
    var buf = momentum_buf[i];
    buf = buf * momentum + g * (1.0 - dampening);
    momentum_buf[i] = buf;
    if (nesterov) {
      update = g + momentum * buf;
    } else {
      update = buf;
    }
  }

  param[i] = p - lr * update;
}

