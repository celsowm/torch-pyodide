import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  MASKED_SELECT_SHADER,
  MASKED_FILL_SHADER,
  createStorageBuffer,
  registerTensor,
  coerceScalarByDType,
  readFromGPU,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class MaskingOps {
  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {}

  async maskedSelect(tensorId: number, maskId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const mask = this.getMeta(maskId);
    if (source.length !== mask.length) {
      throw new Error("maskedSelect requires tensor and mask with same number of elements.");
    }
    if (source.shape.join(",") !== mask.shape.join(",")) {
      throw new Error("maskedSelect requires tensor and mask with same shape.");
    }

    // First pass: count true values on CPU (simplified approach)
    // For a full GPU approach we'd need a prefix sum.
    // Here we use CPU counting + output sizing.
    const maskValues = await this.readTensorValues(maskId, "float32");
    const trueCount = maskValues.filter((v) => v !== 0).length;
    const outLength = trueCount;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const paramsData = new Uint32Array([source.length, outLength]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(MASKED_SELECT_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, mask.buffer, out, paramsBuffer], calculateWorkgroups(source.length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, [outLength], source.dtype, outLength);
    return cloneHandle(meta);
  }

  async maskedFill(tensorId: number, maskId: number, value: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const mask = this.getMeta(maskId);
    if (source.length !== mask.length) {
      throw new Error("maskedFill requires tensor and mask with same number of elements.");
    }
    if (source.shape.join(",") !== mask.shape.join(",")) {
      throw new Error("maskedFill requires tensor and mask with same shape.");
    }

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, source.length * 4));
    const fillValue = coerceScalarByDType(value, source.dtype as any);
    const params = new ArrayBuffer(12);
    const view = new DataView(params);
    view.setFloat32(0, fillValue, true);
    view.setUint32(4, source.length, true);
    view.setUint32(8, 0, true);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: 12,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MASKED_FILL_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, mask.buffer, out, paramsBuffer], calculateWorkgroups(source.length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  private async readTensorValues(tensorId: number, _dtype: string): Promise<number[]> {
    const meta = this.getMeta(tensorId);
    return readFromGPU(this.deviceMgr.device!, meta.buffer, meta.length, meta.dtype as any);
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}
