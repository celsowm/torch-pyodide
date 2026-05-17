struct Dims {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn transpose_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = dims.rows * dims.cols;
    if (idx >= total) {
        return;
    }
    let row = idx / dims.cols;
    let col = idx % dims.cols;
    let out_idx = col * dims.rows + row;
    output[out_idx] = input[idx];
}
