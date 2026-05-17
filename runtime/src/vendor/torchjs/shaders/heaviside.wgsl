@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    if (x < 0.0) {
        result[idx] = 0.0;
    } else if (x > 0.0) {
        result[idx] = 1.0;
    } else {
        result[idx] = values[idx];
    }
}
