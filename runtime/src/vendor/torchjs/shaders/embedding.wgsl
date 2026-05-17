@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_embeddings: u32,
  embedding_dim: u32,
  num_indices: u32,
  _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_indices * params.embedding_dim;
  if (idx >= total) { return; }

  let token_idx = idx / params.embedding_dim;
  let embed_idx = idx % params.embedding_dim;

  let token_id = indices[token_idx];
  // Clamp to valid range
  let safe_token = clamp(u32(token_id), 0u, params.num_embeddings - 1u);

  output[idx] = weight[safe_token * params.embedding_dim + embed_idx];
}
