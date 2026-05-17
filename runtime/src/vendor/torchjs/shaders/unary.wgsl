@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = -input[idx];
}

@compute @workgroup_size(256)
fn abs_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = abs(input[idx]);
}

@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sqrt(input[idx]);
}

@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = exp(input[idx]);
}

@compute @workgroup_size(256)
fn exp2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = exp2(input[idx]);
}

@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log(input[idx]);
}

@compute @workgroup_size(256)
fn log10(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log(input[idx]) / 2.302585092994046;
}

@compute @workgroup_size(256)
fn log2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log2(input[idx]);
}

@compute @workgroup_size(256)
fn log1p(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log(1.0 + input[idx]);
}

@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = tanh(input[idx]);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = max(0.0, input[idx]);
}

@compute @workgroup_size(256)
fn sin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sin(input[idx]);
}

@compute @workgroup_size(256)
fn cos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = cos(input[idx]);
}

@compute @workgroup_size(256)
fn tan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = tan(input[idx]);
}

@compute @workgroup_size(256)
fn acos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = acos(input[idx]);
}

@compute @workgroup_size(256)
fn asin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = asin(input[idx]);
}

@compute @workgroup_size(256)
fn atan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = atan(input[idx]);
}

@compute @workgroup_size(256)
fn cosh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = cosh(input[idx]);
}

@compute @workgroup_size(256)
fn sinh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sinh(input[idx]);
}

@compute @workgroup_size(256)
fn acosh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = log(x + sqrt(x * x - 1.0));
}

@compute @workgroup_size(256)
fn asinh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = log(x + sqrt(x * x + 1.0));
}

@compute @workgroup_size(256)
fn atanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = 0.5 * log((1.0 + x) / (1.0 - x));
}

@compute @workgroup_size(256)
fn ceil_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = ceil(input[idx]);
}

@compute @workgroup_size(256)
fn floor_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = floor(input[idx]);
}

@compute @workgroup_size(256)
fn round_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    // WGSL round() uses round-half-to-even (banker's rounding), matching PyTorch
    result[idx] = round(input[idx]);
}

@compute @workgroup_size(256)
fn trunc_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = trunc(input[idx]);
}

@compute @workgroup_size(256)
fn frac_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let val = input[idx];
    result[idx] = val - trunc(val);
}

@compute @workgroup_size(256)
fn reciprocal_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = 1.0 / input[idx];
}

@compute @workgroup_size(256)
fn rsqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = inverseSqrt(input[idx]);
}

@compute @workgroup_size(256)
fn square_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let val = input[idx];
    result[idx] = val * val;
}

const POS_INF: f32 = 1e30;  // Large finite value approximating infinity
const NEG_INF: f32 = -1e30; // Large negative value

fn erf(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

@compute @workgroup_size(256)
fn erf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = erf(input[idx]);
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    result[idx] = 0.5 * x * (1.0 + erf(x * 0.7071067811865475));
}

@compute @workgroup_size(256)
fn softplus_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = select(log(1.0 + exp(x)), x, x > 20.0);
}

@compute @workgroup_size(256)
fn silu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x * (1.0 / (1.0 + exp(-x)));
}

@compute @workgroup_size(256)
fn mish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    result[idx] = x * tanh(sp);
}

@compute @workgroup_size(256)
fn hardsigmoid_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = clamp(x / 6.0 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn hardswish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x * clamp(x / 6.0 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn softsign_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x / (1.0 + abs(x));
}

@compute @workgroup_size(256)
fn tanhshrink_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x - tanh(x);
}

@compute @workgroup_size(256)
fn isnan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_nan = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
    result[idx] = select(0.0, 1.0, is_nan);
}

@compute @workgroup_size(256)
fn isinf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_inf = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) == 0u);
    result[idx] = select(0.0, 1.0, is_inf);
}

@compute @workgroup_size(256)
fn isfinite_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_nan_or_inf = (bits & 0x7f800000u) == 0x7f800000u;
    result[idx] = select(1.0, 0.0, is_nan_or_inf);
}

@compute @workgroup_size(256)
fn isposinf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_pos_inf = bits == 0x7f800000u;
    result[idx] = select(0.0, 1.0, is_pos_inf);
}

@compute @workgroup_size(256)
fn isneginf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_neg_inf = bits == 0xff800000u;
    result[idx] = select(0.0, 1.0, is_neg_inf);
}

@compute @workgroup_size(256)
fn sign_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sign(input[idx]);
}

@compute @workgroup_size(256)
fn sgn_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0);
}

@compute @workgroup_size(256)
fn erfc_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = 1.0 - erf(input[idx]);
}

@compute @workgroup_size(256)
fn expm1_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = exp(input[idx]) - 1.0;
}

@compute @workgroup_size(256)
fn deg2rad_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = input[idx] * 0.017453292519943295;
}

@compute @workgroup_size(256)
fn rad2deg_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = input[idx] * 57.29577951308232;
}

@compute @workgroup_size(256)
fn logical_not_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = select(1.0, 0.0, input[idx] != 0.0);
}

@compute @workgroup_size(256)
fn i0_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = abs(input[idx]);
    // Modified Bessel function I0 using polynomial approximation
    let t = x / 3.75;
    let t2 = t * t;
    var y: f32;
    if (x < 3.75) {
        // I0(x) = 1 + 3.5156229*t^2 + 3.0899424*t^4 + 1.2067492*t^6 + 0.2659732*t^8 + 0.0360768*t^10 + 0.0045813*t^12
        y = 1.0 + 3.5156229 * t2 + 3.0899424 * t2 * t2 + 1.2067492 * t2 * t2 * t2 + 0.2659732 * t2 * t2 * t2 * t2 + 0.0360768 * t2 * t2 * t2 * t2 * t2 + 0.0045813 * t2 * t2 * t2 * t2 * t2 * t2;
    } else {
        // I0(x) = exp(x) / sqrt(x) * (0.39894228 + 0.01328592/t + 0.00225319/t^2 - 0.00157565/t^3 + 0.00916281/t^4 - 0.02057706/t^5 + 0.02635537/t^6 - 0.01647633/t^7 + 0.00392377/t^8)
        let inv_t = 1.0 / t;
        y = exp(x) * (0.39894228 + 0.01328592 * inv_t + 0.00225319 * inv_t * inv_t - 0.00157565 * inv_t * inv_t * inv_t + 0.00916281 * inv_t * inv_t * inv_t * inv_t - 0.02057706 * inv_t * inv_t * inv_t * inv_t * inv_t + 0.02635537 * inv_t * inv_t * inv_t * inv_t * inv_t * inv_t - 0.01647633 * inv_t * inv_t * inv_t * inv_t * inv_t * inv_t * inv_t + 0.00392377 * inv_t * inv_t * inv_t * inv_t * inv_t * inv_t * inv_t * inv_t) * inverseSqrt(x);
    }
    result[idx] = y;
}

// Lanczos approximation for lgamma
fn lgamma_impl(x: f32) -> f32 {
    // Lanczos coefficients
    let coeffs = array<f32, 7>(76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953005e-5, 0.0);
    var y = x;
    var tmp = x + 5.5;
    tmp = tmp - (x + 0.5) * log(tmp);
    var ser = 1.000000000190015;
    for (var j = 0u; j < 6u; j = j + 1u) {
        y = y + 1.0;
        ser = ser + coeffs[j] / y;
    }
    return -tmp + log(2.5066282746310005 * ser / x);
}

@compute @workgroup_size(256)
fn lgamma_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = lgamma_impl(input[idx]);
}

@compute @workgroup_size(256)
fn digamma_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    // Digamma approximation using asymptotic series
    var y = x;
    var psi = 0.0;
    if (x <= 0.0) {
        // Reflection formula
        if (ceil(x) == x) {
            result[idx] = POS_INF; // +inf for integer <= 0
            return;
        }
        psi = 3.141592653589793 / tan(3.141592653589793 * x);
        y = 1.0 - x;
    }
    if (y < 10.0) {
        // Use recurrence to bring argument above 10
        let n = u32(10.0 - y) + 1u;
        for (var i = 0u; i < n; i = i + 1u) {
            psi = psi - 1.0 / y;
            y = y + 1.0;
        }
    }
    // Asymptotic series for large argument
    psi = psi + log(y) - 0.5 / y;
    let y2 = y * y;
    psi = psi - 1.0 / (12.0 * y2);
    psi = psi + 1.0 / (120.0 * y2 * y2);
    psi = psi - 1.0 / (252.0 * y2 * y2 * y2);
    psi = psi + 1.0 / (240.0 * y2 * y2 * y2 * y2);
    // Apply sign for x <= 0
    if (input[idx] <= 0.0) {
        psi = psi + 3.141592653589793 / tan(3.141592653589793 * input[idx]);
    }
    result[idx] = psi;
}