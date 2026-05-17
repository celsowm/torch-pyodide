struct Dims {
    M: u32,
    K: u32,
    N: u32,
    batch: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE_SIZE: u32 = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;
    var sum: f32 = 0.0;
    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;
    for (var t = 0u; t < numTiles; t++) {
        let aRow = row;
        let aCol = t * TILE_SIZE + localCol;
        if (aRow < dims.M && aCol < dims.K) {
            tileA[localRow][localCol] = A[aRow * dims.K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }
        let bRow = t * TILE_SIZE + localRow;
        let bCol = col;
        if (bRow < dims.K && bCol < dims.N) {
            tileB[localRow][localCol] = B[bRow * dims.N + bCol];
        } else {
            tileB[localRow][localCol] = 0.0;
        }
        workgroupBarrier();
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }
        workgroupBarrier();
    }
    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}

@compute @workgroup_size(256)
fn matmul_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalElements = dims.M * dims.N;
    if (idx >= totalElements) {
        return;
    }
    let row = idx / dims.N;
    let col = idx % dims.N;
    var sum: f32 = 0.0;
    for (var k = 0u; k < dims.K; k++) {
        sum += A[row * dims.K + k] * B[k * dims.N + col];
    }
    C[idx] = sum;
}

@compute @workgroup_size(256)
fn matmul_3d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let totalElements = dims.batch * dims.M * dims.N;
    if (idx >= totalElements) {
        return;
    }
    
    let n = idx % dims.N;
    let m = (idx / dims.N) % dims.M;
    let b = idx / (dims.N * dims.M);
    
    var sum: f32 = 0.0;
    for (var k = 0u; k < dims.K; k++) {
        sum += A[(b * dims.M + m) * dims.K + k] * B[(b * dims.K + k) * dims.N + n];
    }
    C[idx] = sum;
}
