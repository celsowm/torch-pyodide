import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { cloneHandle, product } from "./types.js";
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
    const meta = this.deviceMgr.registerTensor(out, x.shape, x.dtype, length);
    return cloneHandle(meta);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Float32Array([minVal, maxVal, length]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CLAMP_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    const result = this.deviceMgr.registerTensor(out, meta.shape, meta.dtype, length);
    return cloneHandle(result);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(aId);
    const b = this.deviceMgr.getTensorMeta(bId);
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error(`matmul currently supports only 2D tensors, got shapes [${a.shape}] and [${b.shape}].`);
    }
    const [m, k] = a.shape;
    const [k2, n] = b.shape;
    if (k !== k2) throw new Error(`matmul dimension mismatch: [${m},${k}] x [${k2},${n}].`);
    const out = createStorageBuffer(this.deviceMgr.device!, m * n * 4);
    const params = new Uint32Array([m, k, n]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul");
    dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramBuffer], calculateWorkgroups(m * n));
    await syncDevice();
    paramBuffer.destroy();
    const result = this.deviceMgr.registerTensor(out, [m, n], a.dtype, m * n);
    return cloneHandle(result);
  }

  private async elementwise(aId: number, bId: number, op: "add" | "mul" | "sub" | "div_op"): Promise<TensorHandle> {
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
    const meta = this.deviceMgr.registerTensor(out, a.shape, a.dtype, length);
    return cloneHandle(meta);
  }
}
