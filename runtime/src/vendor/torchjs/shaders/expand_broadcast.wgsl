struct Params {
    inShape: vec4<u32>,
    broadcastStrides: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) { return; }
    var remaining = idx;
    var inIdx = 0u;
    var stride = 1u;
    for (var d = 3u; d >= 0u; d = d - 1u) {
        let dimSize = params.inShape[d];
        if (dimSize > 0u) {
            let coord = remaining % dimSize;
            remaining = remaining / dimSize;
            inIdx += coord * params.broadcastStrides[d];
        }
    }
    output[idx] = input[inIdx];
}
