import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { cloneHandle, product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  COMPARE_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";
import { BroadcastOps } from "./broadcastOps.js";

export class CompareOps {
  private broadcastOps: BroadcastOps;

  constructor(private deviceMgr: DeviceManager) {
    this.broadcastOps = new BroadcastOps(deviceMgr);
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

  private async compare(aId: number, bId: number, op: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(aId);
    const b = this.deviceMgr.getTensorMeta(bId);
    if (a.shape.join(",") !== b.shape.join(",")) {
      return this.broadcastOps.elementwiseWithBroadcast(a, b, op as any);
    }
    const length = product(a.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(COMPARE_SHADER, op);
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    const meta = this.deviceMgr.registerTensor(out, a.shape, "bool", length);
    return cloneHandle(meta);
  }
}
