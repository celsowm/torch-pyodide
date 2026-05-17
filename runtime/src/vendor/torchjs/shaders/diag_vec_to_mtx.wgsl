struct Params {
    dim: u32,
    offset: i32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) { return; }
    
    let offset = params.offset;
    let n = params.dim + u32(abs(f32(offset)));
    
    var row: u32;
    var col: u32;
    if (offset >= 0) {
        row = i;
        col = i + u32(offset);
    } else {
        row = i + u32(-offset);
        col = i;
    }
    
    output[row * n + col] = input[i];
}
