// Argmin shader - finds index of min value along the last dimension

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;  // (batch_size, num_elements)

@compute @workgroup_size(256)
fn argmin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    let batch_size = dims.x;
    let num_elements = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let offset = batch_idx * num_elements;

    // Find min value and its index
    var min_val = input[offset];
    var min_idx = 0u;

    for (var i = 1u; i < num_elements; i++) {
        let val = input[offset + i];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    output[batch_idx] = i32(min_idx);
}
