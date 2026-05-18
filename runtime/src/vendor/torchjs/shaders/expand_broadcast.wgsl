struct Params {
    output_shape: vec4<u32>,
    broadcastStrides: vec4<u32>,
    ndim: u32,
    total: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total) { return; }
    var remaining = idx;
    var inIdx = 0u;
    let offset = 4u - params.ndim;

    var out_strides: array<u32, 4>;
    out_strides[3] = 1u;
    if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }
    if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }
    if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }

    for (var d = 0u; d < params.ndim; d++) {
        let dim_idx = offset + d;
        let coord = remaining / out_strides[dim_idx];
        remaining = remaining % out_strides[dim_idx];
        inIdx += coord * params.broadcastStrides[dim_idx];
    }
    output[idx] = input[inIdx];
}
