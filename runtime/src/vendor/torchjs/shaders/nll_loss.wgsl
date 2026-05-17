// NLL Loss shader
// Computes negative log likelihood loss: -input[targets]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec2<u32>;  // (batch_size, num_classes)

@compute @workgroup_size(256)
fn nll_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let batch_size = dims.x;
    let num_classes = dims.y;

    if (idx >= batch_size) {
        return;
    }

    // Get target class for this batch item
    let target_class = targets[idx];

    // Loss is -log_prob[target_class]
    let log_prob = input[idx * num_classes + u32(target_class)];
    output[idx] = -log_prob;
}
