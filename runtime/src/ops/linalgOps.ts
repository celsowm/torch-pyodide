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

    const pipeline = getOrCreatePipeline(CHOLESKY_SHADER, "cholesky_small");
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

    // Initialize pivot buffer with identity permutation [0,1,2,...,n-1] per batch
    {
      const tmp = new Uint32Array(batch * n);
      for (let b = 0; b < batch; b++) {
        for (let i = 0; i < n; i++) {
          tmp[b * n + i] = i;
        }
      }
      const uploadBuf = this.deviceMgr.device!.createBuffer({
        size: tmp.byteLength,
        usage: BufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint32Array(uploadBuf.getMappedRange()).set(tmp);
      uploadBuf.unmap();
      const enc2 = this.deviceMgr.device!.createCommandEncoder();
      enc2.copyBufferToBuffer(uploadBuf, 0, pivotBuf, 0, tmp.byteLength);
      this.deviceMgr.device!.queue.submit([enc2.finish()]);
      uploadBuf.destroy();
    }

    const pivotPipeline = getOrCreatePipeline(LU_SHADER, "lu_pivot");
    const updatePipeline = getOrCreatePipeline(LU_SHADER, "lu_update");

    for (let k = 0; k < n - 1; k++) {
      const params = new Uint32Array([n, batch, k]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: params.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);
      dispatchCompute(pivotPipeline, [aBuf, pivotBuf, paramBuffer], calculateWorkgroups(batch));
      await syncDevice();
      dispatchCompute(updatePipeline, [aBuf, pivotBuf, paramBuffer], calculateWorkgroups(batch * n));
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
    const pipeline = getOrCreatePipeline(TRIANGULAR_SOLVE_SHADER, entrypoint);

    for (let k = 0; k < n; k++) {
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
}
