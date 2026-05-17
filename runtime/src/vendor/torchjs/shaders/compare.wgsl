// Element-wise comparison operations

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Element-wise equality: output[i] = 1.0 if a[i] == b[i], else 0.0
@compute @workgroup_size(256)
fn eq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    output[idx] = select(0.0, 1.0, a[idx] == b[idx]);
}

// Element-wise not equal
@compute @workgroup_size(256)
fn ne(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    output[idx] = select(0.0, 1.0, a[idx] != b[idx]);
}

// Element-wise less than
@compute @workgroup_size(256)
fn lt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    output[idx] = select(0.0, 1.0, a[idx] < b[idx]);
}

// Element-wise less than or equal
@compute @workgroup_size(256)
fn le(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    output[idx] = select(0.0, 1.0, a[idx] <= b[idx]);
}

// Element-wise greater than
@compute @workgroup_size(256)
fn gt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    output[idx] = select(0.0, 1.0, a[idx] > b[idx]);
}

// Element-wise greater than or equal
@compute @workgroup_size(256)
fn ge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    output[idx] = select(0.0, 1.0, a[idx] >= b[idx]);
}

@compute @workgroup_size(256)
fn maximum_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = max(a[idx], b[idx]);
}

@compute @workgroup_size(256)
fn minimum_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = min(a[idx], b[idx]);
}

@compute @workgroup_size(256)
fn fmax_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    // fmax in PyTorch handles NaNs differently (propagates non-NaN if one is NaN)
    // WGSL max() behavior on NaN is same as JS Math.max (propagates NaN)
    // To match fmax: if one is NaN, take the other.
    let va = a[idx];
    let vb = b[idx];
    if (va != va) { output[idx] = vb; }
    else if (vb != vb) { output[idx] = va; }
    else { output[idx] = max(va, vb); }
}

@compute @workgroup_size(256)
fn fmin_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    let va = a[idx];
    let vb = b[idx];
    if (va != va) { output[idx] = vb; }
    else if (vb != vb) { output[idx] = va; }
    else { output[idx] = min(va, vb); }
}
