import { TensorHandle, product, SupportedDType } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  CHOLESKY_SHADER,
  LU_SHADER,
  TRIANGULAR_SOLVE_SHADER,
  JACOBI_SHADER,
  GRAM_SCHMIDT_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class LinalgOps {
  constructor(private deviceMgr: DeviceManager) {}

  async cholesky(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const shape = meta.shape;
    const n = shape.length === 3 ? shape[2] : shape[1];
    const batch = shape.length === 3 ? shape[0] : 1;
    const byteSize = meta.bytes;

    const out = createStorageBuffer(this.deviceMgr.device!, byteSize);
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, byteSize);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const pipeline = await getOrCreatePipeline(CHOLESKY_SHADER, "cholesky_small");
    const params = new Uint32Array([n, batch, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    dispatchCompute(pipeline, [out, paramBuffer], calculateWorkgroups(batch));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, product(meta.shape));
  }

  async lu(tensorId: number): Promise<[TensorHandle, TensorHandle]> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const shape = meta.shape;
    const n = shape.length === 3 ? shape[2] : shape[1];
    const batch = shape.length === 3 ? shape[0] : 1;
    const byteSize = meta.bytes;

    const aBuf = createStorageBuffer(this.deviceMgr.device!, byteSize);
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, aBuf, 0, byteSize);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const pivotBytes = batch * n * 4;
    const pivotBuf = createStorageBuffer(this.deviceMgr.device!, pivotBytes);

    // Initialize pivot buffer with identity permutation [0,1,2,...,n-1] per batch.
    // Stored as f32 to match the runtime's f32 storage/readback convention.
    {
      const tmp = new Float32Array(batch * n);
      for (let b = 0; b < batch; b++) {
        for (let i = 0; i < n; i++) {
          tmp[b * n + i] = i;
        }
      }
      const uploadBuf = this.deviceMgr.device!.createBuffer({
        size: tmp.byteLength,
        usage: BufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      new Float32Array(uploadBuf.getMappedRange()).set(tmp);
      uploadBuf.unmap();
      const enc2 = this.deviceMgr.device!.createCommandEncoder();
      enc2.copyBufferToBuffer(uploadBuf, 0, pivotBuf, 0, tmp.byteLength);
      this.deviceMgr.device!.queue.submit([enc2.finish()]);
      uploadBuf.destroy();
    }

    const pivotPipeline = await getOrCreatePipeline(LU_SHADER, "lu_pivot");
    const scalePipeline = await getOrCreatePipeline(LU_SHADER, "lu_scale");
    const updatePipeline = await getOrCreatePipeline(LU_SHADER, "lu_update");

    for (let k = 0; k < n - 1; k++) {
      const params = new Uint32Array([n, batch, k]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: params.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);
      dispatchCompute(pivotPipeline, [aBuf, pivotBuf, paramBuffer], calculateWorkgroups(batch));
      await syncDevice();
      dispatchCompute(scalePipeline, [aBuf, pivotBuf, paramBuffer], calculateWorkgroups(batch * n));
      await syncDevice();
      dispatchCompute(updatePipeline, [aBuf, pivotBuf, paramBuffer], calculateWorkgroups(batch * n * n));
      await syncDevice();
      paramBuffer.destroy();
    }

    const aHandle = this.deviceMgr.registerTensorAsHandle(aBuf, meta.shape, meta.dtype, product(meta.shape));
    const pivotHandle = this.deviceMgr.registerTensorAsHandle(pivotBuf, [batch, n], "int32" as SupportedDType, batch * n);
    return [aHandle, pivotHandle];
  }

  async triangularSolve(aId: number, bId: number, upper: boolean): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const aMeta = this.deviceMgr.getTensorMeta(aId);
    const bMeta = this.deviceMgr.getTensorMeta(bId);
    const shapeA = aMeta.shape;
    const n = shapeA[shapeA.length - 1];
    const batch = shapeA.length === 3 ? shapeA[0] : 1;
    const m = bMeta.shape[bMeta.shape.length - 1];
    const bSize = bMeta.bytes;

    const bBuf = createStorageBuffer(this.deviceMgr.device!, bSize);
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(bMeta.buffer, 0, bBuf, 0, bSize);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const entrypoint = upper ? "backward_sub_step" : "forward_sub_step";
    const pipeline = await getOrCreatePipeline(TRIANGULAR_SOLVE_SHADER, entrypoint);

    // Forward substitution (lower) processes rows 0..n-1; backward substitution
    // (upper) must process rows n-1..0 so each step only depends on already
    // solved rows.
    for (let step = 0; step < n; step++) {
      const k = upper ? n - 1 - step : step;
      const params = new Uint32Array([n, m, batch, k]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: params.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);
      dispatchCompute(pipeline, [aMeta.buffer, bBuf, paramBuffer], calculateWorkgroups(m * batch));
      await syncDevice();
      paramBuffer.destroy();
    }

    return this.deviceMgr.registerTensorAsHandle(bBuf, bMeta.shape, bMeta.dtype, product(bMeta.shape));
  }

  async jacobiRotate(
    aId: number,
    vId: number,
    p: number,
    q: number,
    c: number,
    s: number,
  ): Promise<[TensorHandle, TensorHandle]> {
    await this.deviceMgr.ensureReady();
    const aMeta = this.deviceMgr.getTensorMeta(aId);
    const vMeta = this.deviceMgr.getTensorMeta(vId);
    const aShape = aMeta.shape;
    const vShape = vMeta.shape;
    const n = aShape[aShape.length - 1];

    const aOut = createStorageBuffer(this.deviceMgr.device!, aMeta.bytes);
    const vOut = createStorageBuffer(this.deviceMgr.device!, vMeta.bytes);

    const pipeline = await getOrCreatePipeline(JACOBI_SHADER, "main");
    const ab = new ArrayBuffer(32);
    const u32 = new Uint32Array(ab);
    const f32 = new Float32Array(ab);
    u32[0] = n;
    u32[1] = p;
    u32[2] = q;
    f32[3] = c;
    f32[4] = s;
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: ab.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, ab);
    dispatchCompute(
      pipeline,
      [aMeta.buffer, aOut, vMeta.buffer, vOut, paramBuffer],
      calculateWorkgroups(n, 64),
    );
    await syncDevice();
    paramBuffer.destroy();

    const aHandle = this.deviceMgr.registerTensorAsHandle(
      aOut,
      aShape,
      aMeta.dtype,
      product(aShape),
    );
    const vHandle = this.deviceMgr.registerTensorAsHandle(
      vOut,
      vShape,
      vMeta.dtype,
      product(vShape),
    );
    return [aHandle, vHandle];
  }

  async gramSchmidt(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    if (meta.shape.length !== 2) {
      throw new Error(`gramSchmidt expects a 2D input, got rank ${meta.shape.length}`);
    }
    const rows = meta.shape[0]!;
    const cols = meta.shape[1]!;
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));

    const pipeline = await getOrCreatePipeline(GRAM_SCHMIDT_SHADER, "main");
    const params = new Uint32Array([rows, cols, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(1));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }
}
