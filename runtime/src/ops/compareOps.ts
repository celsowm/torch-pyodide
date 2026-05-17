import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  COMPARE_SHADER,
  createStorageBuffer,
  registerTensor,
} from "./utils.js";
import { DeviceManager } from "./device.js";
import { BroadcastOps } from "./broadcastOps.js";

type CompareOp = "eq" | "ne" | "lt" | "le" | "gt" | "ge";

export class CompareOps {
  private broadcast: BroadcastOps;

  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {
    this.broadcast = new BroadcastOps(deviceMgr, tensors, nextId, allocatedBytes);
  }

  async eq(aId: number, bId: number): Promise<TensorHandle> {
    return this.compare(aId, bId, "eq");
  }

  async ne(aId: number, bId: number): Promise<TensorHandle> {
    return this.compare(aId, bId, "ne");
  }

  async lt(aId: number, bId: number): Promise<TensorHandle> {
    return this.compare(aId, bId, "lt");
  }

  async le(aId: number, bId: number): Promise<TensorHandle> {
    return this.compare(aId, bId, "le");
  }

  async gt(aId: number, bId: number): Promise<TensorHandle> {
    return this.compare(aId, bId, "gt");
  }

  async ge(aId: number, bId: number): Promise<TensorHandle> {
    return this.compare(aId, bId, "ge");
  }

  private async compare(aId: number, bId: number, op: CompareOp): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.getMeta(aId);
    const b = this.getMeta(bId);

    if (a.shape.join(",") !== b.shape.join(",")) {
      const result = await this.broadcast.elementwiseWithBroadcast(a, b, op as any);
      // Re-cast as bool
      return result;
    }
    if (a.length !== b.length) {
      throw new Error(`Shape mismatch for ${op}: ${a.length} != ${b.length}.`);
    }

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, a.length * 4));
    const pipeline = getOrCreatePipeline(COMPARE_SHADER, op);
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(a.length));
    await syncDevice();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, a.shape, "bool", a.length);
    return cloneHandle(meta);
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}
