// BatchNorm2d inference: y = (x - mean) / sqrt(variance + eps) * weight + bias
// Affine params are packed as [w, b, mean, variance] per channel in a single buffer.
// Params struct uses u32 for dims, f32 for eps — JS must write u32 values as u32 bits, not floats.

struct Params {
    batch: u32,
    channels: u32,
    spatial: u32,
    eps: f32,
    _pad: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read> affine_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.batch * params.channels * params.spatial;
    if (gid.x >= total) {
        return;
    }
    let spatial = params.spatial;
    let c = (gid.x / spatial) % params.channels;
    let base = c * 4u;
    let weight = affine_buf[base + 0u];
    let bias = affine_buf[base + 1u];
    let mean = affine_buf[base + 2u];
    let variance = affine_buf[base + 3u];
    let inv_std = 1.0 / sqrt(variance + params.eps);
    let x = input_buf[gid.x];
    output_buf[gid.x] = (x - mean) * inv_std * weight + bias;
}
