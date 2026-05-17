struct Params {
    batch_size: u32,
    dim_size: u32,
    stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = arrayLength(&output);
    if (idx >= total) { return; }

    let stride_after = params.stride;
    let dim_size = params.dim_size;
    
    let after_idx = idx % stride_after;
    let mid_idx = (idx / stride_after) % dim_size;
    let before_idx = idx / (stride_after * dim_size);
    
    let flipped_mid = dim_size - 1u - mid_idx;
    let input_idx = (before_idx * dim_size + flipped_mid) * stride_after + after_idx;
    
    output[idx] = input[input_idx];
}
