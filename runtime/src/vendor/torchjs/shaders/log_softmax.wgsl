// Log-softmax shader
// Computes log(softmax(x)) along the last dimension

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;  // (batch_size, num_classes)

@compute @workgroup_size(1)
fn log_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    let num_classes = dims.y;

    if (batch_idx >= dims.x) {
        return;
    }

    let offset = batch_idx * num_classes;

    // Find max for numerical stability
    var max_val = input[offset];
    for (var i = 1u; i < num_classes; i++) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute sum of exp(x - max)
    var sum_exp = 0.0;
    for (var i = 0u; i < num_classes; i++) {
        sum_exp += exp(input[offset + i] - max_val);
    }

    // Compute log_softmax = x - max - log(sum_exp)
    let log_sum_exp = log(sum_exp);
    for (var i = 0u; i < num_classes; i++) {
        output[offset + i] = input[offset + i] - max_val - log_sum_exp;
    }
}
