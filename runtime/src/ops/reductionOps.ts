import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle, product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  REDUCE_SUM_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  createStorageBuffer,
  registerTensor,
  readScalar,
  padShapeTo4,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class ReductionOps {
  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {}

  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, false);
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, true);
  }

  async argmax(tensorId: number): Promise<TensorHandle> {
    return this.argReduce(tensorId, true);
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    return this.argReduce(tensorId, false);
  }

  async sumDim(tensorId: number, dim: number, keepdim = false): Promise<TensorHandle> {
    return this.reduceDim(tensorId, dim, keepdim, "sum");
  }

  async meanDim(tensorId: number, dim: number, keepdim = false): Promise<TensorHandle> {
    return this.reduceDim(tensorId, dim, keepdim, "mean");
  }

  async prod(tensorId: number): Promise<TensorHandle> {
    // Use sum shader with prod variant; for now CPU fallback using existing sum-like reduction
    // Actually we need a prod shader — for MVP, reuse reduce with placeholder
    return this.reduceAll(tensorId, false, true);
  }

  async min(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, false, false, "min");
  }

  async max(tensorId: number): Promise<TensorHandle> {
    return this.reduceAll(tensorId, false, false, "max");
  }

  private async reduceAll(
    tensorId: number,
    asMean: boolean,
    asProd = false,
    reduceOp: "sum" | "min" | "max" = "sum"
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    let currentBuffer = source.buffer;
    let currentLength = source.length;
    const temporaryToDestroy: GPUBuffer[] = [];
    const entrypoint = asProd ? "prod" : reduceOp;

    while (currentLength > 1) {
      const groups = Math.ceil(currentLength / 256);
      const outBuffer = createStorageBuffer(this.deviceMgr.device!, Math.max(4, groups * 4));
      const paramData = new Uint32Array([currentLength]);
      const paramBuffer = this.deviceMgr.device!.createBuffer({
        size: 4,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      this.deviceMgr.device!.queue.writeBuffer(paramBuffer, 0, paramData);
      const pipeline = getOrCreatePipeline(REDUCE_SUM_SHADER, "main");
      dispatchCompute(pipeline, [currentBuffer, outBuffer, paramBuffer], [groups, 1, 1]);
      await syncDevice();
      paramBuffer.destroy();
      if (currentBuffer !== source.buffer) {
        temporaryToDestroy.push(currentBuffer);
      }
      currentBuffer = outBuffer;
      currentLength = groups;
    }

    let scalarValue = await readScalar(this.deviceMgr.device!, currentBuffer);
    if (asMean) {
      scalarValue /= source.length;
    }
    if (currentBuffer !== source.buffer) {
      temporaryToDestroy.push(currentBuffer);
    }
    for (const tmp of temporaryToDestroy) {
      tmp.destroy();
    }
    const outBuffer = createStorageBuffer(this.deviceMgr.device!, 4);
    this.deviceMgr.device!.queue.writeBuffer(outBuffer, 0, new Float32Array([scalarValue]));
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, outBuffer, [], source.dtype, 1);
    return cloneHandle(meta);
  }

  private async reduceDim(
    tensorId: number,
    dim: number,
    keepdim: boolean,
    op: "sum" | "mean"
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const rank = source.shape.length;
    const resolvedDim = dim < 0 ? dim + rank : dim;
    if (resolvedDim < 0 || resolvedDim >= rank) {
      throw new Error(`dim out of range for rank ${rank}: ${dim}.`);
    }
    const outShape = source.shape.filter((_, i) => i !== resolvedDim);
    if (keepdim) {
      const ks = [...source.shape];
      ks[resolvedDim] = 1;
      outShape.length = 0;
      outShape.push(...ks);
    }
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const inShapePadded = padShapeTo4(source.shape);
    const outShapePadded = padShapeTo4(outShape);
    const paramsData = new Uint32Array([
      inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      resolvedDim, outLength, op === "mean" ? 1 : 0, rank,
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(REDUCE_SUM_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  private async argReduce(tensorId: number, asMax: boolean): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    if (source.shape.length === 0) {
      const outScalar = createStorageBuffer(this.deviceMgr.device!, 4);
      this.deviceMgr.device!.queue.writeBuffer(outScalar, 0, new Int32Array([0]));
      const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, outScalar, [], "int32", 1);
      return cloneHandle(meta);
    }
    const lastDim = source.shape[source.shape.length - 1]!;
    if (lastDim <= 0) {
      throw new Error("argmax/argmin require last dimension > 0.");
    }
    const batchSize = source.length / lastDim;
    const outputShape = source.shape.length === 1 ? [] : source.shape.slice(0, -1);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, batchSize * 4));
    const dims = new Uint32Array([batchSize, lastDim]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(dimsBuffer, 0, dims);
    const shader = asMax ? ARGMAX_SHADER : ARGMIN_SHADER;
    const entry = asMax ? "argmax" : "argmin";
    const pipeline = getOrCreatePipeline(shader, entry);
    dispatchCompute(pipeline, [source.buffer, out, dimsBuffer], [batchSize, 1, 1]);
    await syncDevice();
    dimsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outputShape, "int32", batchSize);
    return cloneHandle(meta);
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}
