struct Params {
    rows: u32,
    cols: u32,
    offset: i32,
    diag_len: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.diag_len) { return; }
    
    let offset = params.offset;
    var row: u32;
    var col: u32;
    if (offset >= 0) {
        row = i;
        col = i + u32(offset);
    } else {
        row = i + u32(-offset);
        col = i;
    }
    
    output[i] = input[row * params.cols + col];
}
