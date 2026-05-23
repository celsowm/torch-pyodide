// NLL Loss reduced shader for 2D logits [batch, classes].
// reduction_mode: 1=sum, 2=mean

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<i32>;
@group(0) @binding(2) var<storage, read_write> out_scalar: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // [batch, classes, reduction_mode, _]

@compute @workgroup_size(1)
fn nll_loss_reduced(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  let batch = params.x;
  let classes = params.y;
  let reduction_mode = params.z;

  var acc = 0.0;
  var b: u32 = 0u;
  loop {
    if (b >= batch) { break; }
    let t = u32(max(targets[b], 0));
    let target_idx = min(t, classes - 1u);
    let offset = b * classes + target_idx;
    acc = acc - input[offset];
    b = b + 1u;
  }

  if (reduction_mode == 2u && batch > 0u) {
    out_scalar[0] = acc / f32(batch);
  } else {
    out_scalar[0] = acc;
  }
}

