import { TensorHandle, SupportedDType } from "./types.js";
import { product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  MAX_POOL2D_SHADER,
  AVG_POOL2D_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class PoolingOps {
  constructor(private deviceMgr: DeviceManager) {}

  async maxPool2d(
    inputId: number,
    kernelSize: number[],
    stride: number[],
    padding: number[],
    dilation: number[],
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const [batch, ch, inH, inW] = input.shape;
    const kernelH = kernelSize[0];
    const kernelW = kernelSize[1] ?? kernelSize[0];
    const strideH = stride[0] ?? kernelH;
    const strideW = stride[1] ?? stride[0] ?? kernelW;
    const padH = padding[0] ?? 0;
    const padW = padding[1] ?? padding[0] ?? 0;
    const dilH = dilation[0] ?? 1;
    const dilW = dilation[1] ?? dilation[0] ?? 1;
    const outH = Math.floor((inH + 2 * padH - (dilH * (kernelH - 1) + 1)) / strideH + 1);
    const outW = Math.floor((inW + 2 * padW - (dilW * (kernelW - 1) + 1)) / strideW + 1);
    const total = batch * ch * outH * outW;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const params = new Uint32Array([
      batch, ch, inH, inW, outH, outW, kernelH, kernelW,
      strideH, strideW, padH, padW, dilH, dilW,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = await getOrCreatePipeline(MAX_POOL2D_SHADER, "main");
    dispatchCompute(pipeline, [input.buffer, out, paramBuffer], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batch, ch, outH, outW], input.dtype as SupportedDType, total);
  }

  async avgPool2d(
    inputId: number,
    kernelSize: number[],
    stride: number[],
    padding: number[],
    countIncludePad: boolean,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const [batch, ch, inH, inW] = input.shape;
    const kernelH = kernelSize[0];
    const kernelW = kernelSize[1] ?? kernelSize[0];
    const strideH = stride[0] ?? kernelH;
    const strideW = stride[1] ?? stride[0] ?? kernelW;
    const padH = padding[0] ?? 0;
    const padW = padding[1] ?? padding[0] ?? 0;
    const outH = Math.floor((inH + 2 * padH - kernelH) / strideH + 1);
    const outW = Math.floor((inW + 2 * padW - kernelW) / strideW + 1);
    const total = batch * ch * outH * outW;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const params = new Uint32Array([
      batch, ch, inH, inW, outH, outW, kernelH, kernelW,
      strideH, strideW, padH, padW, countIncludePad ? 1 : 0, 0,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = await getOrCreatePipeline(AVG_POOL2D_SHADER, "main");
    dispatchCompute(pipeline, [input.buffer, out, paramBuffer], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batch, ch, outH, outW], input.dtype as SupportedDType, total);
  }
}
