/**
 * Masked select (boolean indexing) operation.
 * Two-pass algorithm:
 * 1. Count: Count non-zero elements in mask
 * 2. Select: Gather elements where mask is true
 */

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  input_size: u32,
  output_size: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

// Prefix sum buffer for computing output indices
@group(0) @binding(4) var<storage, read> prefix_sum: array<u32>;

@compute @workgroup_size(256)
fn masked_select(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.input_size) {
    return;
  }

  // Check if this element is selected
  if (mask[idx] > 0.0) {
    // Get output index from prefix sum
    // prefix_sum[idx] = number of selected elements before this index
    let out_idx = prefix_sum[idx];
    output[out_idx] = input[idx];
  }
}

// Count non-zero elements in mask (reduction)
@group(0) @binding(0) var<storage, read> mask_count: array<f32>;
@group(0) @binding(1) var<storage, read_write> count: array<u32>;

var<workgroup> shared_count: array<u32, 256>;

@compute @workgroup_size(256)
fn count_nonzero(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let gid = global_id.x;
  let lid = local_id.x;

  // Load and convert to 0/1
  if (gid < arrayLength(&mask_count)) {
    shared_count[lid] = select(0u, 1u, mask_count[gid] > 0.0);
  } else {
    shared_count[lid] = 0u;
  }
  workgroupBarrier();

  // Parallel reduction
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      shared_count[lid] += shared_count[lid + s];
    }
    workgroupBarrier();
  }

  // Write workgroup result
  if (lid == 0u) {
    count[global_id.x / 256u] = shared_count[0];
  }
}

// Compute prefix sum for output indices
@group(0) @binding(0) var<storage, read> mask_prefix: array<f32>;
@group(0) @binding(1) var<storage, read_write> prefix: array<u32>;

@compute @workgroup_size(256)
fn compute_prefix(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let size = arrayLength(&mask_prefix);
  if (idx >= size) {
    return;
  }

  // Simple sequential prefix sum (not optimal but works)
  // For production, use parallel scan algorithm
  var sum = 0u;
  for (var i = 0u; i < idx; i++) {
    if (mask_prefix[i] > 0.0) {
      sum += 1u;
    }
  }
  prefix[idx] = sum;
}
