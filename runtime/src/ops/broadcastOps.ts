import { TensorHandle, TensorMeta, product } from "./types.js";
import { cloneHandle } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  ELEMENTWISE_SHADER,
  BROADCAST_SHADER,
  createStorageBuffer,
  registerTensor,
  padShapeTo4,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class BroadcastOps {
  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {}

  async elementwiseWithBroadcast(
    a: TensorMeta,
    b: TensorMeta,
    op: "add" | "mul" | "sub" | "div_op"
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const outShape = this.broadcastShapes(a.shape, b.shape);
    const outLength = product(outShape);

    // Broadcast both inputs to the common shape if needed
    const aExpanded = a.shape.join(",") !== outShape.join(",")
      ? await this.broadcastTensor(a, outShape)
      : a;
    const bExpanded = b.shape.join(",") !== outShape.join(",")
      ? await this.broadcastTensor(b, outShape)
      : b;

    const aBuf = aExpanded.buffer;
    const bBuf = bExpanded.buffer;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    dispatchCompute(pipeline, [aBuf, bBuf, out], calculateWorkgroups(outLength));
    await syncDevice();

    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, a.dtype, outLength);
    return cloneHandle(meta);
  }

  private broadcastShapes(a: number[], b: number[]): number[] {
    const maxRank = Math.max(a.length, b.length);
    const result: number[] = new Array(maxRank);
    for (let i = 0; i < maxRank; i++) {
      const aDim = a.length - maxRank + i >= 0 ? a[a.length - maxRank + i]! : 1;
      const bDim = b.length - maxRank + i >= 0 ? b[b.length - maxRank + i]! : 1;
      if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
        throw new Error(`Shapes [${a}] and [${b}] cannot be broadcast together.`);
      }
      result[i] = Math.max(aDim, bDim);
    }
    return result;
  }

  private async broadcastTensor(tensor: TensorMeta, targetShape: number[]): Promise<TensorMeta> {
    const rankDiff = targetShape.length - tensor.shape.length;
    const paddedShape = [...new Array(rankDiff).fill(1), ...tensor.shape];
    const outLength = product(targetShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const strides = this.computeStrides(paddedShape);
    const broadcastStrides = paddedShape.map((s, i) => (s === 1 ? 0 : strides[i]!));
    const inPadded = padShapeTo4(paddedShape);
    const bsPadded = padShapeTo4(broadcastStrides);

    const paramsData = new Uint32Array([
      inPadded[0], inPadded[1], inPadded[2], inPadded[3],
      bsPadded[0], bsPadded[1], bsPadded[2], bsPadded[3],
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(BROADCAST_SHADER, "main");
    dispatchCompute(pipeline, [tensor.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();

    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, targetShape, tensor.dtype, outLength);
    return meta;
  }

  private computeStrides(shape: number[]): number[] {
    if (shape.length === 0) return [];
    const strides = new Array<number>(shape.length);
    let running = 1;
    for (let i = shape.length - 1; i >= 0; i -= 1) {
      strides[i] = running;
      running *= shape[i]!;
    }
    return strides;
  }
}
