@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] - b[idx];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] * b[idx];
}

@compute @workgroup_size(256)
fn div_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] / b[idx];
}

@compute @workgroup_size(256)
fn atan2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = atan2(a[idx], b[idx]);
}

@compute @workgroup_size(256)
fn hypot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sqrt(a[idx] * a[idx] + b[idx] * b[idx]);
}

@compute @workgroup_size(256)
fn logaddexp(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = a[idx];
    let y = b[idx];
    let max_val = max(x, y);
    result[idx] = max_val + log(exp(x - max_val) + exp(y - max_val));
}

@compute @workgroup_size(256)
fn bitwise_and(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = bitcast<f32>(bitcast<i32>(a[idx]) & bitcast<i32>(b[idx]));
}

@compute @workgroup_size(256)
fn bitwise_or(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = bitcast<f32>(bitcast<i32>(a[idx]) | bitcast<i32>(b[idx]));
}

@compute @workgroup_size(256)
fn bitwise_xor(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = bitcast<f32>(bitcast<i32>(a[idx]) ^ bitcast<i32>(b[idx]));
}

@compute @workgroup_size(256)
fn heaviside(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = select(0.0, 1.0, a[idx] >= 0.0);
}

fn integer_pow(base: f32, exponent: f32) -> f32 {
    let rounded = floor(exponent + 0.5);
    var n = i32(abs(rounded));
    var acc = 1.0;
    var factor = base;
    while (n > 0) {
        if ((n % 2) == 1) {
            acc = acc * factor;
        }
        factor = factor * factor;
        n = n / 2;
    }
    if (rounded < 0.0) {
        return 1.0 / acc;
    }
    return acc;
}

@compute @workgroup_size(256)
fn pow_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let exponent = b[idx];
    let rounded = floor(exponent + 0.5);
    if (abs(exponent - rounded) < 0.000001 && abs(rounded) <= 64.0) {
        result[idx] = integer_pow(a[idx], exponent);
    } else {
        result[idx] = pow(a[idx], exponent);
    }
}
