import { TensorHandle, SupportedDType } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  CONV_SHADER,
  CONV_BACKWARD_SHADER,
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

  async conv2dInputBackward(
    gradOutputId: number,
    weightId: number,
    inputShape: number[],
    gradOutputShape: number[],
    stride: number[],
    padding: number[],
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const weight = this.deviceMgr.getTensorMeta(weightId);
    const [batch, inCh, inH, inW] = inputShape;
    const [, outCh, outH, outW] = gradOutputShape;
    const [, , kernelH, kernelW] = weight.shape;
    const strideH = stride[0] ?? 1;
    const strideW = stride[1] ?? stride[0] ?? 1;
    const padH = padding[0] ?? 0;
    const padW = padding[1] ?? padding[0] ?? 0;
    const groups = 1;
    const total = batch * inCh * inH * inW;

    const gradInput = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));
    // Dummy buffers for bindings not used by this entry point
    const dummyGradWeight = createStorageBuffer(this.deviceMgr.device!, 4);
    const dummyInput = createStorageBuffer(this.deviceMgr.device!, 4);

    const params = new Uint32Array([
      batch, inCh, outCh, inH, inW, outH, outW, kernelH, kernelW,
      strideH, strideW, padH, padW, groups, 0,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = getOrCreatePipeline(CONV_BACKWARD_SHADER, "conv2d_input_backward");
    dispatchCompute(pipeline, [gradOutput.buffer, weight.buffer, gradInput, paramBuffer, dummyGradWeight, dummyInput], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    dummyGradWeight.destroy();
    dummyInput.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradInput, [batch, inCh, inH, inW], gradOutput.dtype as SupportedDType, total);
  }

  async conv2dWeightBackward(
    gradOutputId: number,
    inputId: number,
    weightShape: number[],
    gradOutputShape: number[],
    inputShape: number[],
    stride: number[],
    padding: number[],
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const input = this.deviceMgr.getTensorMeta(inputId);
    const [outCh, inCh, kernelH, kernelW] = weightShape;
    const [batch] = inputShape;
    const [, , outH, outW] = gradOutputShape;
    const strideH = stride[0] ?? 1;
    const strideW = stride[1] ?? stride[0] ?? 1;
    const padH = padding[0] ?? 0;
    const padW = padding[1] ?? padding[0] ?? 0;
    const groups = 1;
    const total = outCh * inCh * kernelH * kernelW;

    const gradWeight = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));
    // Dummy buffers for bindings not used by this entry point
    const dummyGradInput = createStorageBuffer(this.deviceMgr.device!, 4);
    const dummyWeight = createStorageBuffer(this.deviceMgr.device!, 4);

    const params = new Uint32Array([
      inputShape[0], inputShape[1], outCh, inputShape[2], inputShape[3], outH, outW, kernelH, kernelW,
      strideH, strideW, padH, padW, groups, 0,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = getOrCreatePipeline(CONV_BACKWARD_SHADER, "conv2d_weight_backward");
    dispatchCompute(pipeline, [gradOutput.buffer, dummyWeight, dummyGradInput, paramBuffer, gradWeight, input.buffer], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    dummyGradInput.destroy();
    dummyWeight.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradWeight, [outCh, inCh, kernelH, kernelW], gradOutput.dtype as SupportedDType, total);
  }

  async conv2dBiasBackward(
    gradOutputId: number,
    outCh: number,
    gradOutputShape: number[],
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const [batch, , outH, outW] = gradOutputShape;
    const strideH = 1;
    const strideW = 1;
    const padH = 0;
    const padW = 0;
    const groups = 1;
    // Reconstruct inCh/inCh from gradOutput shape (not really used for bias backward, but shader expects them)
    const inCh = 1; // placeholder
    const inH = outH; // placeholder
    const inW = outW; // placeholder
    const total = outCh;

    const gradBias = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));
    // Dummy buffers for bindings not used by this entry point
    const dummyWeight = createStorageBuffer(this.deviceMgr.device!, 4);
    const dummyGradInput = createStorageBuffer(this.deviceMgr.device!, 4);
    const dummyInput = createStorageBuffer(this.deviceMgr.device!, 4);

    const params = new Uint32Array([
      batch, inCh, outCh, inH, inW, outH, outW, 1, 1,
      strideH, strideW, padH, padW, groups, 0,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = getOrCreatePipeline(CONV_BACKWARD_SHADER, "conv2d_bias_backward");
    dispatchCompute(pipeline, [gradOutput.buffer, dummyWeight, dummyGradInput, paramBuffer, gradBias, dummyInput], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    dummyWeight.destroy();
    dummyGradInput.destroy();
    dummyInput.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradBias, [outCh], gradOutput.dtype as SupportedDType, total);
  }
}
