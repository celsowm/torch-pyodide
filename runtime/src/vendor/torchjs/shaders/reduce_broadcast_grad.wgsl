struct Params { 
  batch_size: u32, 
  features: u32 
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let j = global_id.x;
  if (j >= params.features) { return; }

  var sum = 0.0;
  for (var i = 0u; i < params.batch_size; i++) {
    sum += input[i * params.features + j];
  }
  output[j] = sum;
}
