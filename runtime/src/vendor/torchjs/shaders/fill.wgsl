struct Params {
    value: f32,
    length: u32,
}

@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.length) {
        return;
    }
    result[idx] = params.value;
}
