struct Params {
    scalar: f32,
    length: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    result[idx] = input[idx] + params.scalar;
}

@compute @workgroup_size(256)
fn sub_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    result[idx] = input[idx] - params.scalar;
}

@compute @workgroup_size(256)
fn mul_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    result[idx] = input[idx] * params.scalar;
}

@compute @workgroup_size(256)
fn div_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    result[idx] = input[idx] / params.scalar;
}

@compute @workgroup_size(256)
fn pow_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    
    let base = input[idx];
    let exponent = params.scalar;
    
    // Improved precision for integer powers (e.g. 3^2 = 9 exactly)
    let exp_int = i32(round(exponent));
    if (abs(exponent - f32(exp_int)) < 1e-6 && exp_int >= 0 && exp_int <= 10) {
        var res = 1.0;
        for (var i = 0; i < exp_int; i = i + 1) {
            res = res * base;
        }
        result[idx] = res;
        } else {
            result[idx] = pow(base, exponent);
        }
    }
    
    @compute @workgroup_size(256)
    fn eq_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.length) { return; }
        result[idx] = select(0.0, 1.0, input[idx] == params.scalar);
    }
    
    @compute @workgroup_size(256)
    fn ne_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.length) { return; }
        result[idx] = select(0.0, 1.0, input[idx] != params.scalar);
    }
    
    @compute @workgroup_size(256)
    fn lt_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.length) { return; }
        result[idx] = select(0.0, 1.0, input[idx] < params.scalar);
    }
    
    @compute @workgroup_size(256)
    fn le_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.length) { return; }
        result[idx] = select(0.0, 1.0, input[idx] <= params.scalar);
    }
    
    @compute @workgroup_size(256)
    fn gt_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.length) { return; }
        result[idx] = select(0.0, 1.0, input[idx] > params.scalar);
    }
    
    @compute @workgroup_size(256)
    fn ge_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.length) { return; }
        result[idx] = select(0.0, 1.0, input[idx] >= params.scalar);
    }
    