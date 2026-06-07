import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  REDUCE_SUM_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  REDUCE_DIM_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  ELEMENTWISE_SHADER,
  UNARY_SHADER,
  LOG_SOFTMAX_SHADER,
  LOG_SOFTMAX_BACKWARD_SHADER,
  SOFTMAX_BACKWARD_SHADER,
  SOFTMAX_SHADER,
  NLL_LOSS_SHADER,
  NLL_LOSS_BACKWARD_SHADER,
  CROSS_ENTROPY_SHADER,
  CROSS_ENTROPY_BACKWARD_SHADER,
  ADAM_STEP_SHADER,
  ADAMW_STEP_SHADER,
  NLL_LOSS_REDUCED_SHADER,
  SGD_STEP_SHADER,
  RMSPROP_STEP_SHADER,
  MAXMIN_BACKWARD_SHADER,
  EXTENDED_STEP_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

const WORKGROUP_SIZE = 256;

export class ReductionOps {
  constructor(private deviceMgr: DeviceManager) {}

  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, "sum");
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, "mean");
  }

  async sumDim(tensorId: number, dim: number, keepdim: boolean): Promise<TensorHandle> {
    return this.reduceDim(tensorId, dim, keepdim, "sum");
  }

  async meanDim(tensorId: number, dim: number, keepdim: boolean): Promise<TensorHandle> {
    return this.reduceDim(tensorId, dim, keepdim, "mean");
  }

  async prod(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, "prod");
  }

  async min(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, "min");
  }

  async max(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, "max");
  }

  async any(tensorId: number): Promise<TensorHandle> {
    const result = await this.reduceAll(tensorId, "sum");
    return result; // sum > 0 gives "any" — user must check via tolist
  }

  async all(tensorId: number): Promise<TensorHandle> {
    const result = await this.reduceAll(tensorId, "sum");
    return result; // sum == length gives "all"
  }

  async cumsum(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Uint32Array([1, length, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(CUMSUM_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(1));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async cumprod(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Uint32Array([1, length, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(CUMPROD_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(1));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async argmax(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const batchElements = product(meta.shape.slice(0, -1));
    const reduceDim = meta.shape[meta.shape.length - 1]!;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, batchElements * 4));
    const params = new Uint32Array([batchElements, reduceDim, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(ARGMAX_SHADER, "argmax");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(batchElements));
    await syncDevice();
    paramBuffer.destroy();

    const finalShape = meta.shape.slice(0, -1).length > 0 ? meta.shape.slice(0, -1) : [1];
    return this.deviceMgr.registerTensorAsHandle(out, finalShape, "int32", batchElements);
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const batchElements = product(meta.shape.slice(0, -1));
    const reduceDim = meta.shape[meta.shape.length - 1]!;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, batchElements * 4));
    const params = new Uint32Array([batchElements, reduceDim, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(ARGMIN_SHADER, "argmin");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(batchElements));
    await syncDevice();
    paramBuffer.destroy();

    const finalShape = meta.shape.slice(0, -1).length > 0 ? meta.shape.slice(0, -1) : [1];
    return this.deviceMgr.registerTensorAsHandle(out, finalShape, "int32", batchElements);
  }

  private async reduceAll(tensorId: number, mode: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const device = this.deviceMgr.device!;
    let src = meta.buffer;
    let currentLength = product(meta.shape);

    while (currentLength > 1) {
      const groups = Math.ceil(currentLength / WORKGROUP_SIZE);
      const outSize = Math.max(4, groups * 4);
      const out = createStorageBuffer(this.deviceMgr.device!, outSize);
      const params = new Uint32Array([currentLength, 0, 0, 0]);
      const paramBuffer = device.createBuffer({
        size: params.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);

      let pipeline;
      switch (mode) {
        case "prod":
          pipeline = await getOrCreatePipeline(REDUCE_PROD_SHADER, "main");
          break;
        case "max":
          pipeline = await getOrCreatePipeline(REDUCE_MAX_SHADER, "main");
          break;
        case "min":
          pipeline = await getOrCreatePipeline(REDUCE_MIN_SHADER, "main");
          break;
        case "sum":
        case "mean":
        default:
          pipeline = await getOrCreatePipeline(REDUCE_SUM_SHADER, "main");
          break;
      }
      dispatchCompute(pipeline, [src, out, paramBuffer], calculateWorkgroups(groups));

      src = out;
      currentLength = groups;
      paramBuffer.destroy();
    }

    await syncDevice();

    let scalarValue = await this.deviceMgr.readScalar(src);
    if (mode === "mean") {
      scalarValue /= product(meta.shape);
    }
    const scalarBuf = createStorageBuffer(this.deviceMgr.device!, 4);
    this.deviceMgr.writeBuffer(scalarBuf, 0, new Float32Array([scalarValue]));

    return this.deviceMgr.registerTensorAsHandle(scalarBuf, [], meta.dtype, 1);
  }

  private async reduceDim(tensorId: number, dim: number, keepdim: boolean, mode: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const rank = meta.shape.length;
    const resolvedDim = dim < 0 ? dim + rank : dim;
    if (resolvedDim < 0 || resolvedDim >= rank) throw new Error(`dim ${dim} out of range for rank ${rank}`);

    const reduceSize = meta.shape[resolvedDim]!;
    const outShape = meta.shape.filter((_, i) => i !== resolvedDim);
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const batchSize = product(meta.shape.slice(0, resolvedDim));
    const innerSize = product(meta.shape.slice(resolvedDim + 1));
    const op = mode === "mean" ? 1 : 0;
    const params = new Uint32Array([
      batchSize,
      reduceSize,
      innerSize,
      op,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = await getOrCreatePipeline(REDUCE_DIM_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();

    const finalShape = keepdim
      ? meta.shape.map((s, i) => (i === resolvedDim ? 1 : s))
      : outShape;
    return this.deviceMgr.registerTensorAsHandle(out, finalShape, meta.dtype, outLength);
  }

  async nllLoss(inputId: number, targetsId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const targets = this.deviceMgr.getTensorMeta(targetsId);
    const batchSize = targets.length;
    const numClasses = input.shape[input.shape.length - 1];
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, batchSize * 4));
    const params = new Uint32Array([batchSize, numClasses, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(NLL_LOSS_SHADER, "nll_loss");
    dispatchCompute(pipeline, [input.buffer, targets.buffer, out, paramBuffer], calculateWorkgroups(batchSize));
    await syncDevice();
    paramBuffer.destroy();
    const outShape = [batchSize];
    return this.deviceMgr.registerTensorAsHandle(out, outShape, input.dtype as SupportedDType, batchSize);
  }

  async nllLossReduced(inputId: number, targetsId: number, reduction: "sum" | "mean"): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const targets = this.deviceMgr.getTensorMeta(targetsId);
    const batchSize = targets.length;
    const numClasses = input.shape[input.shape.length - 1]!;
    if (batchSize * numClasses !== input.length) {
      throw new Error(
        `nllLossReduced expects input shape [..., C] and target shape [...]; got input length ${input.length}, target length ${batchSize}, classes ${numClasses}`,
      );
    }
    const out = createStorageBuffer(this.deviceMgr.device!, 4);
    const reductionMode = reduction === "sum" ? 1 : 2;
    const params = new Uint32Array([batchSize, numClasses, reductionMode, 0]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramsBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(NLL_LOSS_REDUCED_SHADER, "nll_loss_reduced");
    dispatchCompute(pipeline, [input.buffer, targets.buffer, out, paramsBuffer], [1]);
    await syncDevice();
    paramsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [], input.dtype as SupportedDType, 1);
  }

  async softmax(tensorId: number, dim: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const resolvedDim = dim < 0 ? dim + meta.shape.length : dim;
    if (resolvedDim < 0 || resolvedDim >= meta.shape.length) {
      throw new Error(`dim ${dim} out of range for rank ${meta.shape.length}`);
    }

    let outer = 1;
    for (let i = 0; i < resolvedDim; i++) outer *= meta.shape[i]!;
    const axisSize = meta.shape[resolvedDim]!;
    let inner = 1;
    for (let i = resolvedDim + 1; i < meta.shape.length; i++) inner *= meta.shape[i]!;
    const rows = outer * inner;
    const outLength = product(meta.shape);

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const dims = new Uint32Array([rows, axisSize, inner, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    const pipeline = await getOrCreatePipeline(SOFTMAX_SHADER, "softmax");
    dispatchCompute(pipeline, [meta.buffer, out, dimsBuffer], calculateWorkgroups(rows));
    await syncDevice();
    dimsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype as SupportedDType, outLength);
  }

  async logSoftmax(tensorId: number, dim: number): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const resolvedDim = dim < 0 ? dim + meta.shape.length : dim;
    const length = product(meta.shape);

    let outer = 1;
    for (let i = 0; i < resolvedDim; i++) outer *= meta.shape[i]!;
    const axisSize = meta.shape[resolvedDim]!;
    let inner = 1;
    for (let i = resolvedDim + 1; i < meta.shape.length; i++) inner *= meta.shape[i]!;
    const batchSize = outer * inner;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, meta.bytes);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    // Shader expects dims = (batch_size, num_classes); log_softmax.wgsl
    // uses var<uniform> dims: vec2<u32>.
    const params = new Uint32Array([batchSize, axisSize, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: Math.max(16, params.byteLength),
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(LOG_SOFTMAX_SHADER, "log_softmax");
    // Shader uses @workgroup_size(1), so dispatch one workgroup per batch.
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], [batchSize, 1, 1]);
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async logSoftmaxBackward(
    gradOutputId: number,
    softmaxId: number,
    batchSize: number,
    numClasses: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const total = batchSize * numClasses;

    const gradInput = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const dims = new Uint32Array([batchSize, numClasses, 0, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);

    const pipeline = await getOrCreatePipeline(LOG_SOFTMAX_BACKWARD_SHADER, "log_softmax_backward");
    dispatchCompute(pipeline, [gradOutput.buffer, this.deviceMgr.getTensorMeta(softmaxId).buffer, gradInput, dimsBuffer], calculateWorkgroups(total));
    await syncDevice();
    dimsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradInput, [batchSize, numClasses], gradOutput.dtype as SupportedDType, total);
  }

  async softmaxBackward(
    gradOutputId: number,
    softmaxId: number,
    batchSize: number,
    numClasses: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const total = batchSize * numClasses;

    const gradInput = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));
    const dims = new Uint32Array([batchSize, numClasses, 0, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);

    const pipeline = await getOrCreatePipeline(SOFTMAX_BACKWARD_SHADER, "softmax_backward");
    dispatchCompute(
      pipeline,
      [gradOutput.buffer, this.deviceMgr.getTensorMeta(softmaxId).buffer, gradInput, dimsBuffer],
      calculateWorkgroups(total),
    );
    await syncDevice();
    dimsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradInput, [batchSize, numClasses], gradOutput.dtype as SupportedDType, total);
  }

  async nllLossBackward(
    targetsId: number,
    batchSize: number,
    numClasses: number,
    scale: number = 1.0,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const targets = this.deviceMgr.getTensorMeta(targetsId);
    const total = batchSize * numClasses;

    const gradInput = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const params = new Float32Array([batchSize, numClasses, scale, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = await getOrCreatePipeline(NLL_LOSS_BACKWARD_SHADER, "nll_loss_backward");
    dispatchCompute(pipeline, [targets.buffer, gradInput, paramBuffer], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradInput, [batchSize, numClasses], "float32", total);
  }

  async crossEntropy(inputId: number, targetsId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const targets = this.deviceMgr.getTensorMeta(targetsId);
    const batchSize = targets.length;
    const numClasses = input.shape[input.shape.length - 1]!;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, batchSize * 4));
    const dims = new Uint32Array([batchSize, numClasses, 0, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    const pipeline = await getOrCreatePipeline(CROSS_ENTROPY_SHADER, "cross_entropy");
    dispatchCompute(pipeline, [input.buffer, targets.buffer, out, dimsBuffer], calculateWorkgroups(batchSize));
    await syncDevice();
    dimsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batchSize], input.dtype as SupportedDType, batchSize);
  }

  async crossEntropyBackward(
    gradOutputId: number,
    inputId: number,
    targetsId: number,
    reduction: "none" | "sum" | "mean",
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const input = this.deviceMgr.getTensorMeta(inputId);
    const targets = this.deviceMgr.getTensorMeta(targetsId);
    const batchSize = targets.length;
    const numClasses = input.shape[input.shape.length - 1]!;
    const total = batchSize * numClasses;
    const reductionMode = reduction === "none" ? 0 : reduction === "sum" ? 1 : 2;
    const gradIsScalar = gradOutput.length === 1 ? 1 : 0;
    const normScale = reduction === "mean" ? 1.0 / Math.max(1, batchSize) : 1.0;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));
    const dims = new Uint32Array([batchSize, numClasses, reductionMode, gradIsScalar]);
    const scales = new Float32Array([normScale, 0, 0, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    const scalesBuffer = this.deviceMgr.device!.createBuffer({
      size: scales.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(scalesBuffer, 0, scales);
    const pipeline = await getOrCreatePipeline(CROSS_ENTROPY_BACKWARD_SHADER, "cross_entropy_backward");
    dispatchCompute(
      pipeline,
      [gradOutput.buffer, input.buffer, targets.buffer, out, dimsBuffer, scalesBuffer],
      calculateWorkgroups(total),
    );
    await syncDevice();
    dimsBuffer.destroy();
    scalesBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batchSize, numClasses], input.dtype as SupportedDType, total);
  }

  async adamStep(
    paramId: number,
    gradId: number,
    expAvgId: number,
    expAvgSqId: number,
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    stepSize: number,
    invSqrtBiasCorrection2: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const expAvg = this.deviceMgr.getTensorMeta(expAvgId);
    const expAvgSq = this.deviceMgr.getTensorMeta(expAvgSqId);
    const n = param.length;
    if (grad.length !== n || expAvg.length !== n || expAvgSq.length !== n) {
      throw new Error("adamStep: tensor lengths must match");
    }

    const dims = new Uint32Array([n, 0, 0, 0]);
    const hp = new Float32Array([lr, beta1, beta2, eps]);
    const extra = new Float32Array([weightDecay, stepSize, invSqrtBiasCorrection2, 0]);

    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);

    const pipeline = await getOrCreatePipeline(ADAM_STEP_SHADER, "adam_step");
    dispatchCompute(
      pipeline,
      [param.buffer, grad.buffer, expAvg.buffer, expAvgSq.buffer, dimsBuffer, hpBuffer, extraBuffer],
      calculateWorkgroups(n),
    );
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async adamWStep(
    paramId: number,
    gradId: number,
    expAvgId: number,
    expAvgSqId: number,
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    stepSize: number,
    invSqrtBiasCorrection2: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const expAvg = this.deviceMgr.getTensorMeta(expAvgId);
    const expAvgSq = this.deviceMgr.getTensorMeta(expAvgSqId);
    const n = param.length;
    if (grad.length !== n || expAvg.length !== n || expAvgSq.length !== n) {
      throw new Error("adamWStep: tensor lengths must match");
    }

    const dims = new Uint32Array([n, 0, 0, 0]);
    const hp = new Float32Array([lr, beta1, beta2, eps]);
    const extra = new Float32Array([weightDecay, stepSize, invSqrtBiasCorrection2, 0]);

    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);

    const pipeline = await getOrCreatePipeline(ADAMW_STEP_SHADER, "adamw_step");
    dispatchCompute(
      pipeline,
      [param.buffer, grad.buffer, expAvg.buffer, expAvgSq.buffer, dimsBuffer, hpBuffer, extraBuffer],
      calculateWorkgroups(n),
    );
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async sgdStep(
    paramId: number,
    gradId: number,
    momentumBufId: number,
    lr: number,
    momentum: number,
    weightDecay: number,
    dampening: number,
    nesterov: boolean,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const momentumBuf = this.deviceMgr.getTensorMeta(momentumBufId);
    const n = param.length;
    if (grad.length !== n || momentumBuf.length !== n) throw new Error("sgdStep: tensor lengths must match");
    const hasMomentum = momentum !== 0 ? 1 : 0;
    const dims = new Uint32Array([n, hasMomentum, nesterov ? 1 : 0, 0]);
    const hp = new Float32Array([lr, momentum, weightDecay, dampening]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    const pipeline = await getOrCreatePipeline(SGD_STEP_SHADER, "sgd_step");
    dispatchCompute(pipeline, [param.buffer, grad.buffer, momentumBuf.buffer, dimsBuffer, hpBuffer], calculateWorkgroups(n));
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
  }

  async rmspropStep(
    paramId: number,
    gradId: number,
    squareAvgId: number,
    momentumBufId: number,
    lr: number,
    alpha: number,
    eps: number,
    weightDecay: number,
    momentum: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const squareAvg = this.deviceMgr.getTensorMeta(squareAvgId);
    const momentumBuf = this.deviceMgr.getTensorMeta(momentumBufId);
    const n = param.length;
    if (grad.length !== n || squareAvg.length !== n || momentumBuf.length !== n) throw new Error("rmspropStep: tensor lengths must match");
    const dims = new Uint32Array([n, momentum !== 0 ? 1 : 0, 0, 0]);
    const hp = new Float32Array([lr, alpha, eps, weightDecay]);
    const extra = new Float32Array([momentum, 0, 0, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);
    const pipeline = await getOrCreatePipeline(RMSPROP_STEP_SHADER, "rmsprop_step");
    dispatchCompute(pipeline, [param.buffer, grad.buffer, squareAvg.buffer, momentumBuf.buffer, dimsBuffer, hpBuffer, extraBuffer], calculateWorkgroups(n));
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async adagradStep(
    paramId: number,
    gradId: number,
    sumSquaresId: number,
    lr: number,
    eps: number,
    weightDecay: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const sumSquares = this.deviceMgr.getTensorMeta(sumSquaresId);
    const n = param.length;
    if (grad.length !== n || sumSquares.length !== n) {
      throw new Error("adagradStep: tensor lengths must match");
    }
    const dims = new Uint32Array([n, 0, 0, 0]);
    const hp = new Float32Array([lr, 0, 0, eps]);
    const extra = new Float32Array([weightDecay, 0, 0, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);
    const pipeline = await getOrCreatePipeline(EXTENDED_STEP_SHADER, "adagrad_step");
    // Adagrad's entrypoint only references bindings 0, 1, 2, 4, 5, 6
    // (state1 @binding(3) is declared but unused in the function body,
    // so the auto-generated bind group layout omits it). We pass an
    // explicit entries array instead of a sequential buffer list to
    // avoid "binding index 3 not present in the bind group layout".
    dispatchCompute(
      pipeline,
      [
        { binding: 0, resource: { buffer: param.buffer } },
        { binding: 1, resource: { buffer: grad.buffer } },
        { binding: 2, resource: { buffer: sumSquares.buffer } },
        { binding: 4, resource: { buffer: dimsBuffer } },
        { binding: 5, resource: { buffer: hpBuffer } },
        { binding: 6, resource: { buffer: extraBuffer } },
      ],
      calculateWorkgroups(n),
    );
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async adamaxStep(
    paramId: number,
    gradId: number,
    expAvgId: number,
    expInfId: number,
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    stepSize: number,
    biasCorrection1: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const expAvg = this.deviceMgr.getTensorMeta(expAvgId);
    const expInf = this.deviceMgr.getTensorMeta(expInfId);
    const n = param.length;
    if (grad.length !== n || expAvg.length !== n || expInf.length !== n) {
      throw new Error("adamaxStep: tensor lengths must match");
    }
    const dims = new Uint32Array([n, 0, 0, 0]);
    const hp = new Float32Array([lr, beta1, beta2, eps]);
    const extra = new Float32Array([weightDecay, stepSize, biasCorrection1, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);
    const pipeline = await getOrCreatePipeline(EXTENDED_STEP_SHADER, "adamax_step");
    dispatchCompute(
      pipeline,
      [param.buffer, grad.buffer, expAvg.buffer, expInf.buffer, dimsBuffer, hpBuffer, extraBuffer],
      calculateWorkgroups(n),
    );
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async nadamStep(
    paramId: number,
    gradId: number,
    expAvgId: number,
    expAvgSqId: number,
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    stepSize: number,
    mu: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const expAvg = this.deviceMgr.getTensorMeta(expAvgId);
    const expAvgSq = this.deviceMgr.getTensorMeta(expAvgSqId);
    const n = param.length;
    if (grad.length !== n || expAvg.length !== n || expAvgSq.length !== n) {
      throw new Error("nadamStep: tensor lengths must match");
    }
    const dims = new Uint32Array([n, 0, 0, 0]);
    const hp = new Float32Array([lr, beta1, beta2, eps]);
    const extra = new Float32Array([weightDecay, stepSize, mu, 0]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);
    const pipeline = await getOrCreatePipeline(EXTENDED_STEP_SHADER, "nadam_step");
    dispatchCompute(
      pipeline,
      [param.buffer, grad.buffer, expAvg.buffer, expAvgSq.buffer, dimsBuffer, hpBuffer, extraBuffer],
      calculateWorkgroups(n),
    );
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async radamStep(
    paramId: number,
    gradId: number,
    expAvgId: number,
    expAvgSqId: number,
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    stepSize: number,
    beta1PowT: number,
    beta2PowT: number,
    stepCount: number,
  ): Promise<void> {
    await this.deviceMgr.ensureReady();
    const param = this.deviceMgr.getTensorMeta(paramId);
    const grad = this.deviceMgr.getTensorMeta(gradId);
    const expAvg = this.deviceMgr.getTensorMeta(expAvgId);
    const expAvgSq = this.deviceMgr.getTensorMeta(expAvgSqId);
    const n = param.length;
    if (grad.length !== n || expAvg.length !== n || expAvgSq.length !== n) {
      throw new Error("radamStep: tensor lengths must match");
    }
    const dims = new Uint32Array([n, stepCount >>> 0, 0, 0]);
    const hp = new Float32Array([lr, beta1, beta2, eps]);
    const extra = new Float32Array([weightDecay, stepSize, beta1PowT, beta2PowT]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({ size: dims.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const hpBuffer = this.deviceMgr.device!.createBuffer({ size: hp.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    const extraBuffer = this.deviceMgr.device!.createBuffer({ size: extra.byteLength, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.deviceMgr.writeBuffer(dimsBuffer, 0, dims);
    this.deviceMgr.writeBuffer(hpBuffer, 0, hp);
    this.deviceMgr.writeBuffer(extraBuffer, 0, extra);
    const pipeline = await getOrCreatePipeline(EXTENDED_STEP_SHADER, "radam_step");
    dispatchCompute(
      pipeline,
      [param.buffer, grad.buffer, expAvg.buffer, expAvgSq.buffer, dimsBuffer, hpBuffer, extraBuffer],
      calculateWorkgroups(n),
    );
    await syncDevice();
    dimsBuffer.destroy();
    hpBuffer.destroy();
    extraBuffer.destroy();
  }

  async maxMinBackward(inputId: number, gradOutputId: number, mode: "max" | "min"): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const n = input.length;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, n * 4));
    const params = new Uint32Array([n, mode === "max" ? 0 : 1, 0, 0]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramsBuffer, 0, params);
    const pipeline = await getOrCreatePipeline(MAXMIN_BACKWARD_SHADER, "maxmin_backward");
    dispatchCompute(pipeline, [input.buffer, gradOutput.buffer, out, paramsBuffer], [1]);
    await syncDevice();
    paramsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, input.shape, input.dtype as SupportedDType, n);
  }

  private async elementwiseOp(aId: number, bIdOrScalar: number, op: string): Promise<TensorHandle> {
    const a = this.deviceMgr.getTensorMeta(aId);
    const aLength = product(a.shape);
    const bIsId = bIdOrScalar >= 100;
    if (bIsId) {
      const b = this.deviceMgr.getTensorMeta(bIdOrScalar);
      const bLength = product(b.shape);
      const outLength = Math.max(aLength, bLength);
      const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
      const pipeline = await getOrCreatePipeline(ELEMENTWISE_SHADER, op);
      dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(outLength));
      await syncDevice();
      return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, outLength);
    } else {
      const scalar = bIdOrScalar;
      const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, aLength * 4));
      const params = new Float32Array([scalar, aLength, 0, 0]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);
      const pipeline = await getOrCreatePipeline(UNARY_SHADER, op);
      dispatchCompute(pipeline, [a.buffer, out], calculateWorkgroups(aLength));
      await syncDevice();
      paramBuffer.destroy();
      return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, aLength);
    }
  }
}
