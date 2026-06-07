import { TensorHandle, SupportedDType } from "./types.js";
import { product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  EMBEDDING_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class EmbeddingOps {
  constructor(private deviceMgr: DeviceManager) {}

  async embedding(
    weightId: number,
    indicesId: number,
    numEmbeddings: number,
    embeddingDim: number,
    paddingIdx: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const weight = this.deviceMgr.getTensorMeta(weightId);
    const indices = this.deviceMgr.getTensorMeta(indicesId);
    const numIndices = product(indices.shape);
    const total = numIndices * embeddingDim;

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const params = new Uint32Array([
      numEmbeddings >>> 0,
      embeddingDim >>> 0,
      numIndices >>> 0,
      paddingIdx >= 0 ? paddingIdx >>> 0 : 0xFFFFFFFF,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);

    const pipeline = await getOrCreatePipeline(EMBEDDING_SHADER, "main");
    dispatchCompute(
      pipeline,
      [weight.buffer, indices.buffer, out, paramBuffer],
      calculateWorkgroups(total),
    );
    await syncDevice();
    paramBuffer.destroy();

    const outShape = [...indices.shape, embeddingDim];
    return this.deviceMgr.registerTensorAsHandle(
      out,
      outShape,
      weight.dtype as SupportedDType,
      total,
    );
  }
}
