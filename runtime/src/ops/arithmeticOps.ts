import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  ELEMENTWISE_SHADER,
  WHERE_SHADER,
  MATMUL_SHADER,
  CLAMP_SHADER,
  createStorageBuffer,
  registerTensor,
} from "./utils.js";
import { DeviceManager } from "./device.js";
import { BroadcastOps } from "./broadcastOps.js";

export class ArithmeticOps {
  private broadcast: BroadcastOps;

  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {
    this.broadcast = new BroadcastOps(deviceMgr, tensors, nextId, allocatedBytes);
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
    const condition = this.getMeta(conditionId);
    const x = this.getMeta(xId);
    const y = this.getMeta(yId);
    if (condition.length !== x.length || x.length !== y.length) {
      throw new Error("where requires condition, x and y with same number of elements.");
    }
    if (condition.shape.join(",") !== x.shape.join(",") || x.shape.join(",") !== y.shape.join(",")) {
      throw new Error("where requires condition, x and y with same shape.");
    }
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, x.length * 4));
    const pipeline = getOrCreatePipeline(WHERE_SHADER, "main");
    dispatchCompute(pipeline, [condition.buffer, x.buffer, y.buffer, out], calculateWorkgroups(x.length));
    await syncDevice();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, x.shape, x.dtype, x.length);
    return cloneHandle(meta);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, source.length * 4));
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setFloat32(0, minVal, true);
    view.setFloat32(4, maxVal, true);
    view.setUint32(8, source.length, true);
    view.setUint32(12, 0, true);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CLAMP_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(source.length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.getMeta(aId);
    const b = this.getMeta(bId);
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("matmul currently supports only 2D tensors.");
    }
    const [m, kA] = a.shape;
    const [kB, n] = b.shape;
    if (kA !== kB) {
      throw new Error(`matmul dimension mismatch: ${kA} != ${kB}.`);
    }
    const outLength = m * n;
    const outBuffer = createStorageBuffer(this.deviceMgr.device!, outLength * 4);
    const dimsData = new Uint32Array([m, kA, n, 1]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dimsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul_2d");
    dispatchCompute(pipeline, [a.buffer, b.buffer, outBuffer, dimsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    dimsBuffer.destroy();
    const out = registerTensor(this.tensors, this.nextId, this.allocatedBytes, outBuffer, [m, n], "float32", outLength);
    return cloneHandle(out);
  }

  private async elementwise(aId: number, bId: number, op: "add" | "mul" | "sub" | "div_op"): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.getMeta(aId);
    const b = this.getMeta(bId);
    // Try broadcasting if shapes differ
    if (a.shape.join(",") !== b.shape.join(",")) {
      return this.broadcast.elementwiseWithBroadcast(a, b, op);
    }
    if (a.length !== b.length) {
      throw new Error(`Shape mismatch for ${op}: ${a.length} != ${b.length}.`);
    }
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, a.length * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(a.length));
    await syncDevice();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, a.shape, a.dtype, a.length);
    return cloneHandle(meta);
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}
