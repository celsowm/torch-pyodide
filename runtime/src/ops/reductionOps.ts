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
  NLL_LOSS_SHADER,
  NLL_LOSS_BACKWARD_SHADER,
  createStorageBuffer,
  padShapeTo4,
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
    const params = new Uint32Array([length, 0, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CUMSUM_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async cumprod(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Uint32Array([length, 0, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CUMPROD_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
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
    const params = new Uint32Array([reduceDim, batchElements, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(ARGMAX_SHADER, "argmax");
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
    const params = new Uint32Array([reduceDim, batchElements, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(ARGMIN_SHADER, "argmin");
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
          pipeline = getOrCreatePipeline(REDUCE_PROD_SHADER, "main");
          break;
        case "max":
          pipeline = getOrCreatePipeline(REDUCE_MAX_SHADER, "main");
          break;
        case "min":
          pipeline = getOrCreatePipeline(REDUCE_MIN_SHADER, "main");
          break;
        case "sum":
        case "mean":
        default:
          pipeline = getOrCreatePipeline(REDUCE_SUM_SHADER, "main");
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
    const inShape = padShapeTo4(meta.shape);
    const inStrides: number[] = [];
    let running = 1;
    for (let i = inShape.length - 1; i >= 0; i--) { inStrides[i] = running; running *= inShape[i]!; }

    const reduceDim = resolvedDim + (4 - rank);
    const reduceSize = meta.shape[resolvedDim]!;
    const outShape = meta.shape.filter((_, i) => i !== resolvedDim);
    if (outShape.length === 0) outShape.push(1);
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const params = new Uint32Array([
      inShape[0], inShape[1], inShape[2], inShape[3],
      inStrides[0]!, inStrides[1]!, inStrides[2]!, inStrides[3]!,
      reduceDim, reduceSize, outLength,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = getOrCreatePipeline(REDUCE_DIM_SHADER, "main");
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
    const pipeline = getOrCreatePipeline(NLL_LOSS_SHADER, "nll_loss");
    dispatchCompute(pipeline, [input.buffer, targets.buffer, out, paramBuffer], calculateWorkgroups(batchSize));
    await syncDevice();
    paramBuffer.destroy();
    const outShape = [batchSize];
    return this.deviceMgr.registerTensorAsHandle(out, outShape, input.dtype as SupportedDType, batchSize);
  }

  async softmax(tensorId: number, dim: number): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const resolvedDim = dim < 0 ? dim + meta.shape.length : dim;
    const maxReduce = await this.reduceDim(tensorId, resolvedDim, true, "max");
    const shifted = await this.elementwiseOp(meta, maxReduce.id, "sub_op");
    const expTensor = await this.elementwiseOp(shifted.id, -1, "exp_op");
    const sumReduce = await this.reduceDim(expTensor.id, resolvedDim, true, "sum");
    const result = await this.elementwiseOp(expTensor.id, sumReduce.id, "div_op");
    maxReduce.destroy?.(); shifted.destroy?.(); expTensor.destroy?.(); sumReduce.destroy?.();
    return result;
  }

  async logSoftmax(tensorId: number, dim: number): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const resolvedDim = dim < 0 ? dim + meta.shape.length : dim;
    const length = product(meta.shape);

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, meta.bytes);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const params = new Int32Array([resolvedDim, meta.shape.length, length, 0,
      ...meta.shape.slice(0, 4).map(s => Math.max(1, s)),
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: Math.max(16, Math.ceil(params.byteLength / 16) * 16),
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(LOG_SOFTMAX_SHADER, "log_softmax");
      dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
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

    const pipeline = getOrCreatePipeline(LOG_SOFTMAX_BACKWARD_SHADER, "log_softmax_backward");
    dispatchCompute(pipeline, [gradOutput.buffer, this.deviceMgr.getTensorMeta(softmaxId).buffer, gradInput, dimsBuffer], calculateWorkgroups(total));
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

    const pipeline = getOrCreatePipeline(NLL_LOSS_BACKWARD_SHADER, "nll_loss_backward");
    dispatchCompute(pipeline, [targets.buffer, gradInput, paramBuffer], calculateWorkgroups(total));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradInput, [batchSize, numClasses], "float32", total);
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
      const params = new Int32Array([aLength, bLength, outLength, 0]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);
      const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
      dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramBuffer], calculateWorkgroups(outLength));
      await syncDevice();
      paramBuffer.destroy();
      return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, outLength);
    } else {
      const scalar = bIdOrScalar;
      const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, aLength * 4));
      const params = new Float32Array([scalar, aLength, 0, 0]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);
      const pipeline = getOrCreatePipeline(UNARY_SHADER, op);
      dispatchCompute(pipeline, [a.buffer, out], calculateWorkgroups(aLength));
      await syncDevice();
      paramBuffer.destroy();
      return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, aLength);
    }
  }
}
