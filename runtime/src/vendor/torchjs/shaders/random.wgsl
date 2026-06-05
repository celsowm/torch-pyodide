struct RNGParams {
    seed: u32,
    length: u32,
    param0: f32,
    param1: f32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: RNGParams;

fn xorshift(state: u32) -> u32 {
    var x = state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

fn uint_to_float(x: u32) -> f32 {
    return f32(x) / 4294967296.0;
}

@compute @workgroup_size(256)
fn rand(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    var state = params.seed ^ (idx * 1664525u + 1013904223u);
    state = xorshift(state);
    output[idx] = uint_to_float(state);
}

@compute @workgroup_size(256)
fn randn(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    var state1 = params.seed ^ (idx * 1664525u + 1013904223u);
    state1 = xorshift(state1);
    var state2 = xorshift(state1);
    let u1 = uint_to_float(state1);
    let u2 = uint_to_float(state2);
    let r = sqrt(-2.0 * log(max(u1, 0.0000001)));
    let theta = 6.283185307179586 * u2;
    output[idx] = r * cos(theta);
}

@compute @workgroup_size(256)
fn normal_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // param0 = mean, param1 = std
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    var state1 = params.seed ^ (idx * 1664525u + 1013904223u);
    state1 = xorshift(state1);
    var state2 = xorshift(state1);
    let u1 = uint_to_float(state1);
    let u2 = uint_to_float(state2);
    let r = sqrt(-2.0 * log(max(u1, 0.0000001)));
    let theta = 6.283185307179586 * u2;
    output[idx] = params.param0 + params.param1 * r * cos(theta);
}

@compute @workgroup_size(256)
fn bernoulli_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // param0 = probability p
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    var state = params.seed ^ (idx * 1664525u + 1013904223u);
    state = xorshift(state);
    let u = uint_to_float(state);
    output[idx] = select(0.0, 1.0, u < params.param0);
}

@compute @workgroup_size(256)
fn exponential_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // param0 = rate (lambda), 1/lambda * (-ln(u))
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    var state = params.seed ^ (idx * 1664525u + 1013904223u);
    state = xorshift(state);
    let u = uint_to_float(state);
    // Clamp u to avoid log(0)
    let u_clamped = max(u, 1e-7);
    output[idx] = -log(u_clamped) / params.param0;
}

@compute @workgroup_size(256)
fn log_normal_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // param0 = mean (of underlying normal), param1 = std (of underlying normal)
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    var state1 = params.seed ^ (idx * 1664525u + 1013904223u);
    state1 = xorshift(state1);
    var state2 = xorshift(state1);
    let u1 = uint_to_float(state1);
    let u2 = uint_to_float(state2);
    let r = sqrt(-2.0 * log(max(u1, 0.0000001)));
    let theta = 6.283185307179586 * u2;
    output[idx] = exp(params.param0 + params.param1 * r * cos(theta));
}
