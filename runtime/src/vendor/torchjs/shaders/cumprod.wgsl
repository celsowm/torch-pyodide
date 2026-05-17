struct Params {
    batch_size: u32,
    reduce_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= params.batch_size) { return; }

    let offset = batch_idx * params.reduce_size;
    var acc: f32 = 1.0;
    for (var i = 0u; i < params.reduce_size; i++) {
        acc *= input[offset + i];
        output[offset + i] = acc;
    }
}
