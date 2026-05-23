// TopK backward scatter.
// indices are per-segment positions in the original input along dim.

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct Params {
  k: u32,
  num_segs: u32,
  seg_stride: u32,
  outer_stride_in: u32,
  outer_stride_out: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn topk_backward(@builtin(local_invocation_id) lid: vec3<u32>,
                 @builtin(workgroup_id) wg_id: vec3<u32>) {
  let seg = wg_id.x;
  if (seg >= params.num_segs) { return; }

  let pos = lid.x;
  if (pos >= params.k) { return; }

  let src = seg * params.outer_stride_out + pos * params.seg_stride;
  let dst_pos = u32(max(0.0, indices[src]));
  let dst = seg * params.outer_stride_in + dst_pos * params.seg_stride;
  grad_input[dst] = grad_output[src];
}

