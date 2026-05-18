import { TensorHandle, SupportedDType } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  CONV_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class ConvOps {
  constructor(private deviceMgr: DeviceManager) {}

  async conv2d(
    inputId: number,
    weightId: number,
    bias: number[] | null,
    stride: number[],
    padding: number[],
    dilation: number[],
    groups: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const w = this.deviceMgr.getTensorMeta(weightId);
    const [batch, inCh, inH, inW] = input.shape;
    const [outCh, , kernelH, kernelW] = w.shape;
    const padH = padding[0] ?? 0;
    const padW = padding[1] ?? padding[0] ?? 0;
    const strideH = stride[0] ?? 1;
    const strideW = stride[1] ?? stride[0] ?? 1;
    const outH = Math.floor((inH + 2 * padH - kernelH) / strideH + 1);
    const outW = Math.floor((inW + 2 * padW - kernelW) / strideW + 1);
    const total = batch * outCh * outH * outW;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    // If bias is null, create a dummy zero bias so the shader can still bind it
    const biasData = bias ?? new Array(outCh).fill(0);
    const biasBuf = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outCh * 4));
    this.deviceMgr.writeBuffer(biasBuf, 0, new Float32Array(biasData));

    const params = new Uint32Array([
      batch, inCh, outCh, inH, inW, outH, outW, kernelH, kernelW,
      strideH, strideW, padH, padW, groups, 0,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = getOrCreatePipeline(CONV_SHADER, "conv2d");
    dispatchCompute(pipeline, [input.buffer, w.buffer, biasBuf, out, paramBuffer], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    biasBuf.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batch, outCh, outH, outW], input.dtype as SupportedDType, total);
  }
}
