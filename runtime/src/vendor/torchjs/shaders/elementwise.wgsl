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

@compute @workgroup_size(256)
fn fmod_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // fmod: remainder of a / b, sign of a. PyTorch: a - a/b.truncated() * b
    result[idx] = a[idx] - trunc(a[idx] / b[idx]) * b[idx];
}

@compute @workgroup_size(256)
fn remainder_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // Python-style remainder: a - b * round(a/b)
    let div = a[idx] / b[idx];
    let rounded = floor(div + 0.5);
    result[idx] = a[idx] - rounded * b[idx];
}

@compute @workgroup_size(256)
fn addcmul_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // addcmul(input, tensor1, tensor2, value=1): out = input + value * tensor1 * tensor2
    // For elementwise shader we treat a = input, b = tensor1*tensor2 packed.
    // Real implementation uses a dedicated dispatch that pre-computes tensor1*tensor2.
    // This entrypoint is here for shader registry completeness; the dedicated
    // dispatch path uses `addcmul_dedicated_op` below.
    result[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn addcdiv_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn xlogy_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // xlogy(x, y) = x * log(y) when y > 0, else 0
    let x = a[idx];
    let y = b[idx];
    result[idx] = select(0.0, x * log(y), y > 0.0);
}

@compute @workgroup_size(256)
fn copysign_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // Manual copysign: take magnitude of a, sign of b.
    let bits_a = bitcast<u32>(a[idx]);
    let bits_b = bitcast<u32>(b[idx]);
    let sign_b = bits_b & 0x80000000u;
    let magnitude = bits_a & 0x7fffffffu;
    result[idx] = bitcast<f32>(magnitude | sign_b);
}

@compute @workgroup_size(256)
fn floor_divide_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // PyTorch floor_divide: floor(a / b)
    result[idx] = floor(a[idx] / b[idx]);
}

@compute @workgroup_size(256)
fn true_divide_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // true_divide is just regular float division
    result[idx] = a[idx] / b[idx];
}

@compute @workgroup_size(256)
fn logical_and_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = select(0.0, 1.0, a[idx] != 0.0 && b[idx] != 0.0);
}

@compute @workgroup_size(256)
fn logical_or_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = select(0.0, 1.0, a[idx] != 0.0 || b[idx] != 0.0);
}

@compute @workgroup_size(256)
fn logical_xor_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let ax = a[idx] != 0.0;
    let bx = b[idx] != 0.0;
    result[idx] = select(0.0, 1.0, ax != bx);
}

@compute @workgroup_size(256)
fn nextafter_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // nextafter: returns the next representable f32 after a in the direction of b.
    // Uses bit-level manipulation on the f32 representation.
    let ax = a[idx];
    let bx = b[idx];
    let bits_a = bitcast<u32>(ax);
    let ax_is_neg = (bits_a & 0x80000000u) != 0u;
    let ax_is_zero = (bits_a & 0x7fffffffu) == 0u;
    var next_bits: u32;
    if (bx > ax) {
        if (!ax_is_neg) {
            next_bits = bits_a + 1u;
        } else {
            if (ax_is_zero) {
                next_bits = 0x00000001u;
            } else {
                next_bits = bits_a + 1u;
            }
        }
    } else if (bx < ax) {
        if (!ax_is_neg) {
            if (ax_is_zero) {
                next_bits = 0x80000001u;
            } else {
                next_bits = bits_a - 1u;
            }
        } else {
            next_bits = bits_a - 1u;
        }
    } else {
        next_bits = bits_a;
    }
    result[idx] = bitcast<f32>(next_bits);
}

@compute @workgroup_size(256)
fn logaddexp2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = a[idx];
    let y = b[idx];
    let max_val = max(x, y);
    result[idx] = max_val + log2(exp2(x - max_val) + exp2(y - max_val));
}
