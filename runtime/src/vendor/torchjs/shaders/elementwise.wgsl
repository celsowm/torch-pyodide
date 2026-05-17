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
