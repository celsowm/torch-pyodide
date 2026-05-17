import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { cloneHandle, product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  MASKED_SELECT_SHADER,
  MASKED_FILL_SHADER,
  createStorageBuffer,
  coerceScalarByDType,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class MaskingOps {
  constructor(private deviceMgr: DeviceManager) {}

  async maskedSelect(tensorId: number, maskId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const mask = this.deviceMgr.getTensorMeta(maskId);
    const length = product(meta.shape);
    const maskData = await this.deviceMgr.readFromGPU(mask.buffer, length, "bool");
    const trueCount = maskData.filter(v => v !== 0).length;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, trueCount * 4));
    const params = new Uint32Array([length, trueCount]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MASKED_SELECT_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, mask.buffer, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    const result = this.deviceMgr.registerTensor(out, [trueCount], meta.dtype, trueCount);
    return cloneHandle(result);
  }

  async maskedFill(tensorId: number, maskId: number, value: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const mask = this.deviceMgr.getTensorMeta(maskId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, meta.bytes);

    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, meta.bytes);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const fillValue = coerceScalarByDType(value, meta.dtype as SupportedDType);
    const params = new Float32Array([fillValue, length]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MASKED_FILL_SHADER, "main");
    dispatchCompute(pipeline, [out, mask.buffer, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    const result = this.deviceMgr.registerTensor(out, meta.shape, meta.dtype, length);
    return cloneHandle(result);
  }
}
