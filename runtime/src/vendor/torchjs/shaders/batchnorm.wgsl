struct Params {
    batch: u32,
    channels: u32,
    spatial: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> inv_std: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // For (N, C, L) -> total = N * C * L
    // For (N, C, H, W) -> total = N * C * H * W
    let total = params.batch * params.channels * params.spatial;
    if (idx >= total) { return; }

    let spatial_idx = idx % params.spatial;
    let c = (idx / params.spatial) % params.channels;
    let b = idx / (params.channels * params.spatial);

    let inp_idx = ((b * params.channels + c) * params.spatial) + spatial_idx;
    let normalized = (input[inp_idx] - mean[c]) * inv_std[c];
    output[inp_idx] = weight[c] * normalized + bias[c];
}
