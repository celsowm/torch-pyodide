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

  async argmax(tensorId: number): Promise<TensorHandle> {
    return this.argReduce(tensorId, ARGMAX_SHADER, "argmax");
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    return this.argReduce(tensorId, ARGMIN_SHADER, "argmin");
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
      const params = new Uint32Array([currentLength]);
      const paramBuffer = device.createBuffer({
        size: params.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      this.deviceMgr.writeBuffer(paramBuffer, 0, params);

      const shader = this.shaderForMode(mode);
      const pipeline = getOrCreatePipeline(shader, mode);
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

    return this.deviceMgr.registerTensorAsHandle(scalarBuf, [1], meta.dtype, 1);
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

    const shader = REDUCE_DIM_SHADER;
    const pipeline = getOrCreatePipeline(shader, mode);
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();

    const finalShape = keepdim
      ? meta.shape.map((s, i) => (i === resolvedDim ? 1 : s))
      : outShape;
    return this.deviceMgr.registerTensorAsHandle(out, finalShape, meta.dtype, outLength);
  }

  private async argReduce(tensorId: number, shader: string, entrypoint: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const batchElements = product(meta.shape.slice(0, -1));
    const reduceDim = meta.shape[meta.shape.length - 1]!;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, batchElements * 4));
    const params = new Uint32Array([reduceDim, batchElements]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(shader, entrypoint);
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(batchElements));
    await syncDevice();
    paramBuffer.destroy();

    const finalShape = meta.shape.slice(0, -1).length > 0 ? meta.shape.slice(0, -1) : [1];
    return this.deviceMgr.registerTensorAsHandle(out, finalShape, "int32", batchElements);
  }

  private shaderForMode(mode: string): string {
    if (mode === "sum") return REDUCE_SUM_SHADER;
    if (mode === "prod") return REDUCE_PROD_SHADER;
    if (mode === "max") return REDUCE_MAX_SHADER;
    if (mode === "min") return REDUCE_MIN_SHADER;
    if (mode === "mean") return REDUCE_SUM_SHADER;
    return REDUCE_SUM_SHADER;
  }
}
