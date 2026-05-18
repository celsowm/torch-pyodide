import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { product } from "./types.js";
import {
  assertDType,
  coerceScalarByDType,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  ELEMENTWISE_SHADER,
  WHERE_SHADER,
  MATMUL_SHADER,
  CLAMP_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";
import { BroadcastOps } from "./broadcastOps.js";

export class ArithmeticOps {
  private broadcastOps: BroadcastOps;

  constructor(private deviceMgr: DeviceManager) {
    this.broadcastOps = new BroadcastOps(deviceMgr);
  }

  async add(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "add");
  }

  async mul(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "mul");
  }

  async sub(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "sub");
  }

  async div(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "div_op");
  }

  async where(conditionId: number, xId: number, yId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const c = this.deviceMgr.getTensorMeta(conditionId);
    const x = this.deviceMgr.getTensorMeta(xId);
    const y = this.deviceMgr.getTensorMeta(yId);
    if (c.shape.join(",") !== x.shape.join(",") || c.shape.join(",") !== y.shape.join(",")) {
      throw new Error("where requires all tensors to have the same shape.");
    }
    const length = product(c.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(WHERE_SHADER, "main");
    dispatchCompute(pipeline, [c.buffer, x.buffer, y.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, x.shape, x.dtype, length);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Float32Array([minVal, maxVal, length, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CLAMP_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async pow(aId: number, bId: number): Promise<TensorHandle> {
    if (aId !== bId) {
      return this.elementwise(aId, bId, "pow_op");
    }
    // Same tensor — pre-compute? just use elementwise anyway
    return this.elementwise(aId, bId, "pow_op");
  }

  async heaviside(inputId: number, valuesId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(inputId);
    const b = this.deviceMgr.getTensorMeta(valuesId);
    if (a.shape.join(",") !== b.shape.join(",")) {
      throw new Error("heaviside requires same shape for both inputs.");
    }
    const length = product(a.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, "heaviside");
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, length);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(aId);
    const b = this.deviceMgr.getTensorMeta(bId);

    // mv: (m, k) @ (k,) -> (m,)
    if (a.shape.length === 2 && b.shape.length === 1) {
      const [m, k] = a.shape;
      const [k2] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${m},${k}] x [${k2}].`);
      return this._runMatmul2d(m, k, 1, a.buffer, b.buffer, a.dtype);
    }

    // mv: (k,) @ (k, n) -> (n,)
    if (a.shape.length === 1 && b.shape.length === 2) {
      const [k] = a.shape;
      const [k2, n] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${k}] x [${k2},${n}].`);
      return this._runMatmul2d(1, k, n, a.buffer, b.buffer, b.dtype);
    }

    // bmm: (b, m, k) @ (b, k, n) -> (b, m, n)
    if (a.shape.length === 3 && b.shape.length === 3) {
      const [batch, m, k] = a.shape;
      const [batch2, k2, n] = b.shape;
      if (batch !== batch2) throw new Error(`matmul batch dim mismatch: ${batch} vs ${batch2}.`);
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${batch},${m},${k}] x [${batch2},${k2},${n}].`);
      return this._runMatmul3d(batch, m, k, n, a.buffer, b.buffer, a.dtype);
    }

    // mm: (m, k) @ (k, n) -> (m, n)
    if (a.shape.length === 2 && b.shape.length === 2) {
      const [m, k] = a.shape;
      const [k2, n] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${m},${k}] x [${k2},${n}].`);
      return this._runMatmul2d(m, k, n, a.buffer, b.buffer, a.dtype);
    }

    // bmm broadcast: (..., m, k) @ (k, n) -> (..., m, n)
    if (a.shape.length >= 3 && b.shape.length === 2) {
      const batch = a.shape.slice(0, -2).reduce((x, y) => x * y, 1);
      const [m, k] = a.shape.slice(-2);
      const [k2, n] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [...,${m},${k}] x [${k2},${n}].`);
      return this._runMatmul3d(batch, m, k, n, a.buffer, b.buffer, a.dtype);
    }

    throw new Error(`matmul unsupported shapes: [${a.shape}] and [${b.shape}].`);
  }

  private async _runMatmul2d(m: number, k: number, n: number,
    aBuf: GPUBuffer, bBuf: GPUBuffer, dtype: string): Promise<TensorHandle> {
    const out = createStorageBuffer(this.deviceMgr.device!, m * n * 4);
    const params = new Uint32Array([m, k, n, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul_2d");
    dispatchCompute(pipeline, [aBuf, bBuf, out, paramBuffer], calculateWorkgroups(m * n));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [m, n], dtype, m * n);
  }

  private async _runMatmul3d(batch: number, m: number, k: number, n: number,
    aBuf: GPUBuffer, bBuf: GPUBuffer, dtype: string): Promise<TensorHandle> {
    const out = createStorageBuffer(this.deviceMgr.device!, batch * m * n * 4);
    const params = new Uint32Array([m, k, n, batch]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul_3d");
    dispatchCompute(pipeline, [aBuf, bBuf, out, paramBuffer], calculateWorkgroups(batch * m * n));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batch, m, n], dtype, batch * m * n);
  }

  private async elementwise(aId: number, bId: number, op: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(aId);
    const b = this.deviceMgr.getTensorMeta(bId     );
    if (a.shape.join(",") !== b.shape.join(",")) {
      return this.broadcastOps.elementwiseWithBroadcast(a, b, op);
    }
    const length = product(a.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, length);
  }
}
