// Bitonic sort along a specified dimension.
// Each workgroup sorts one "segment" (all elements along the given dim for a fixed position in other dims).
// Uses strides to access non-contiguous segments.
// Output: sorted values + original indices (as f32, cast to i64 on readback).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<f32>;

struct Params {
  seg_size: u32,
  num_segs: u32,
  seg_stride: u32,
  outer_stride: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_vals: array<f32, 2048>;
var<workgroup> shared_idx: array<f32, 2048>;

fn next_pow2(x: u32) -> u32 {
  var v = x - 1u;
  v = v | (v >> 1u);
  v = v | (v >> 2u);
  v = v | (v >> 4u);
  v = v | (v >> 8u);
  v = v | (v >> 16u);
  return v + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {
  let seg = wg_id.x;
  if (seg >= params.num_segs) { return; }

  let seg_size = params.seg_size;
  let seg_stride = params.seg_stride;
  let outer_stride = params.outer_stride;
  // Base offset of this segment: decompose the segment index into the
  // coordinates outside the sorted dimension. For the last dim this reduces
  // to seg * outer_stride, but for inner dims the segments are interleaved.
  let seg_base = (seg / seg_stride) * outer_stride + (seg % seg_stride);
  let pad_size = max(2u, next_pow2(seg_size));
  let tid = lid.x;
  let total_threads = 256u;
  let work_per_thread = (pad_size + total_threads - 1u) / total_threads;

  for (var i = 0u; i < work_per_thread; i += 1u) {
    let pos = tid * work_per_thread + i;
    if (pos < seg_size) {
      let linear_idx = seg_base + pos * seg_stride;
      shared_vals[pos] = input[linear_idx];
      shared_idx[pos] = f32(pos);
    } else if (pos < pad_size) {
      shared_vals[pos] = 3.402823466e+38; // FLT_MAX
      shared_idx[pos] = f32(pos);
    }
  }
  workgroupBarrier();

  var k: u32 = 2u;
  while (k <= pad_size) {
    var j: u32 = k / 2u;
    while (j > 0u) {
      for (var i = 0u; i < work_per_thread; i += 1u) {
        let pos = tid * work_per_thread + i;
        if (pos < pad_size) {
          let ixj = pos ^ j;
          if (ixj > pos && ixj < pad_size) {
            let is_ascending = (pos & k) == 0u;
            if (is_ascending) {
              if (shared_vals[pos] > shared_vals[ixj]) {
                let tmp_v = shared_vals[pos];
                let tmp_i = shared_idx[pos];
                shared_vals[pos] = shared_vals[ixj];
                shared_idx[pos] = shared_idx[ixj];
                shared_vals[ixj] = tmp_v;
                shared_idx[ixj] = tmp_i;
              }
            } else {
              if (shared_vals[pos] < shared_vals[ixj]) {
                let tmp_v = shared_vals[pos];
                let tmp_i = shared_idx[pos];
                shared_vals[pos] = shared_vals[ixj];
                shared_idx[pos] = shared_idx[ixj];
                shared_vals[ixj] = tmp_v;
                shared_idx[ixj] = tmp_i;
              }
            }
          }
        }
      }
      workgroupBarrier();
      j = j / 2u;
    }
    k = k * 2u;
  }

  for (var i = 0u; i < work_per_thread; i += 1u) {
    let pos = tid * work_per_thread + i;
    if (pos < seg_size) {
      let linear_idx = seg_base + pos * seg_stride;
      values[linear_idx] = shared_vals[pos];
      indices[linear_idx] = shared_idx[pos];
    }
  }
}
