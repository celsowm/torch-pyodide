struct Params {
    shape: vec4<u32>,
    strides: vec4<u32>,
    flip_mask: vec4<u32>, // 1 if dimension should be flipped, 0 otherwise
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = arrayLength(&output);
    if (idx >= total) { return; }

    var input_idx = 0u;
    var remaining = idx;
    
    for (var i = 0u; i < 4u; i++) {
        let coord = remaining / params.strides[i];
        remaining = remaining % params.strides[i];
        
        var flipped_coord = coord;
        if (params.flip_mask[i] == 1u) {
            flipped_coord = params.shape[i] - 1u - coord;
        }
        
        input_idx = input_idx + flipped_coord * params.strides[i];
    }
    
    output[idx] = input[input_idx];
}
