struct Params {
    shape: vec4<u32>,
    strides: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }

    var input_idx = 0u;
    var current_idx = idx;
    
    // Extract coordinates in order: d3, d2, d1, d0
    for (var i = 3i; i >= 0; i--) {
        let dim_size = params.shape[i];
        // With right-aligned padding, unused leading dims are 1, so this works.
        let coord = current_idx % dim_size;
        current_idx /= dim_size;
        input_idx += coord * params.strides[i];
    }
    
    output[idx] = input[input_idx];
}
