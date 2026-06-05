import { TensorHandle, TensorMeta } from "./types.js";
import { product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  ELEMENTWISE_SHADER,
  COMPARE_SHADER,
  EXPAND_BROADCAST_SHADER,
  createStorageBuffer,
  padShapeTo4,
  computeStrides,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class BroadcastOps {
  constructor(private deviceMgr: DeviceManager) {}

  async elementwiseWithBroadcast(
    a: TensorMeta,
    b: TensorMeta,
    op: string
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const outShape = this.broadcastShapes(a.shape, b.shape);
    const outLength = product(outShape);

    const aExpanded = a.shape.join(",") !== outShape.join(",")
      ? await this.broadcastTensor(a, outShape)
      : a;
    const bExpanded = b.shape.join(",") !== outShape.join(",")
      ? await this.broadcastTensor(b, outShape)
      : b;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    dispatchCompute(pipeline, [aExpanded.buffer, bExpanded.buffer, out], calculateWorkgroups(outLength));
    await syncDevice();

    return this.deviceMgr.registerTensorAsHandle(out, outShape, a.dtype, outLength);
  }

  async compareWithBroadcast(
    a: TensorMeta,
    b: TensorMeta,
    op: string
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const outShape = this.broadcastShapes(a.shape, b.shape);
    const outLength = product(outShape);

    const aExpanded = a.shape.join(",") !== outShape.join(",")
      ? await this.broadcastTensor(a, outShape)
      : a;
    const bExpanded = b.shape.join(",") !== outShape.join(",")
      ? await this.broadcastTensor(b, outShape)
      : b;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const pipeline = getOrCreatePipeline(COMPARE_SHADER, op);
    dispatchCompute(pipeline, [aExpanded.buffer, bExpanded.buffer, out], calculateWorkgroups(outLength));
    await syncDevice();

    const outDtype = (op === "maximum_op" || op === "minimum_op") ? a.dtype : "bool";
    return this.deviceMgr.registerTensorAsHandle(out, outShape, outDtype, outLength);
  }

  broadcastShapes(a: number[], b: number[]): number[] {
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

  async broadcastTensor(tensor: TensorMeta, targetShape: number[]): Promise<TensorMeta> {
    const rankDiff = targetShape.length - tensor.shape.length;
    const paddedShape = [...new Array(rankDiff).fill(1), ...tensor.shape];
    const outLength = product(targetShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const outShapePadded = padShapeTo4(targetShape);
    const paddedShape4 = padShapeTo4(paddedShape);
    const strides4 = computeStrides(paddedShape4);
    const bsPadded = new Uint32Array(paddedShape4.map((s, i) => (s === 1 ? 0 : strides4[i]!)));

    const paramsData = new Uint32Array([
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      bsPadded[0], bsPadded[1], bsPadded[2], bsPadded[3],
      targetShape.length, outLength, 0, 0,
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(EXPAND_BROADCAST_SHADER, "main");
    dispatchCompute(pipeline, [tensor.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();

    return { ...tensor, buffer: out, shape: targetShape, length: outLength, bytes: out.size };
  }
}
