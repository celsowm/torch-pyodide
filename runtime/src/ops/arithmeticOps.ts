import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { product } from "./types.js";
import {
  assertDType,
  coerceScalarByDType,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  ELEMENTWISE_SHADER,
  TERNARY_SHADER,
  WHERE_SHADER,
  MATMUL_SHADER,
  CLAMP_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";
import { BroadcastOps } from "./broadcastOps.js";
import { UnaryOps } from "./unaryOps.js";

export class ArithmeticOps {
  private broadcastOps: BroadcastOps;
  private unaryOps: UnaryOps;

  constructor(private deviceMgr: DeviceManager) {
    this.broadcastOps = new BroadcastOps(deviceMgr);
    this.unaryOps = new UnaryOps(deviceMgr);
  }

  async add(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "add");
  }

  async mul(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "mul");
  }

  async sub(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "sub");
  }

  async div(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "div_op");
  }

  async atan2(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "atan2_op");
  }

  async hypot(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "hypot_op");
  }

  async logaddexp(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "logaddexp");
  }

  async logaddexp2(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "logaddexp2_op");
  }

  async fmod(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "fmod_op");
  }

  async remainder(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "remainder_op");
  }

  async xlogy(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "xlogy_op");
  }

  async copysign(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "copysign_op");
  }

  async floorDivide(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "floor_divide_op");
  }

  async trueDivide(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "true_divide_op");
  }

  async nextafter(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "nextafter_op");
  }

  async logicalAnd(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "logical_and_op");
  }

  async logicalOr(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "logical_or_op");
  }

  async logicalXor(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "logical_xor_op");
  }

  async bitwiseAnd(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "bitwise_and");
  }

  async bitwiseOr(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "bitwise_or");
  }

  async bitwiseXor(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "bitwise_xor");
  }

  async bitwiseNot(aId: number): Promise<TensorHandle> {
    return this.unaryOps.bitwiseNot(aId);
  }

  /**
   * lerp(start, end, weight) = start + weight * (end - start)
   * Implemented as 3 elementwise dispatches (sub, mul_scalar, add) for the
   * common scalar-weight case.
   */
  async lerpScalar(startId: number, endId: number, weight: number): Promise<TensorHandle> {
    const diff = await this.sub(endId, startId);
    const scaled = await this.mulScalar(diff.id, weight);
    return this.add(startId, scaled.id);
  }

  /**
   * lerp(start, end, weight) where weight is a tensor (broadcasted).
   */
  async lerpTensor(startId: number, endId: number, weightId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(startId);
    const b = this.deviceMgr.getTensorMeta(endId);
    const c = this.deviceMgr.getTensorMeta(weightId);
    // Broadcast a, b, c to the common shape
    const outShape = this.broadcastOps.broadcastShapes(
      this.broadcastOps.broadcastShapes(a.shape, b.shape),
      c.shape,
    );
    const length = product(outShape);
    const aExpanded = a.shape.join(",") !== outShape.join(",")
      ? await this.broadcastOps.broadcastTensor(a, outShape)
      : a;
    const bExpanded = b.shape.join(",") !== outShape.join(",")
      ? await this.broadcastOps.broadcastTensor(b, outShape)
      : b;
    const cExpanded = c.shape.join(",") !== outShape.join(",")
      ? await this.broadcastOps.broadcastTensor(c, outShape)
      : c;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(TERNARY_SHADER, "lerp_op");
    dispatchCompute(pipeline, [aExpanded.buffer, bExpanded.buffer, cExpanded.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, a.dtype, length);
  }

  /**
   * addcmul(input, t1, t2, value=1) = input + value * t1 * t2
   * For value=1 (most common), uses 2 dispatches: mul(t1,t2) + add(input, _).
   * For value != 1, decompose into mul + mul_scalar + add.
   */
  async addcmul(inputId: number, t1Id: number, t2Id: number, value: number): Promise<TensorHandle> {
    const product = await this.mul(t1Id, t2Id);
    if (value === 1.0) {
      return this.add(inputId, product.id);
    }
    const scaled = await this.mulScalar(product.id, value);
    return this.add(inputId, scaled.id);
  }

  /**
   * addcdiv(input, t1, t2, value=1) = input + value * (t1 / t2)
   */
  async addcdiv(inputId: number, t1Id: number, t2Id: number, value: number): Promise<TensorHandle> {
    const quotient = await this.div(t1Id, t2Id);
    if (value === 1.0) {
      return this.add(inputId, quotient.id);
    }
    const scaled = await this.mulScalar(quotient.id, value);
    return this.add(inputId, scaled.id);
  }

  /**
   * mul by a scalar value (broadcasts the scalar across the tensor).
   */
  async mulScalar(tensorId: number, value: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(a.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Float32Array([value, length, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, "mul");
    dispatchCompute(pipeline, [a.buffer, paramBuffer, out], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, length);
  }

  async where(conditionId: number, xId: number, yId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const c = this.deviceMgr.getTensorMeta(conditionId);
    const x = this.deviceMgr.getTensorMeta(xId);
    const y = this.deviceMgr.getTensorMeta(yId);
    if (c.shape.join(",") !== x.shape.join(",") || c.shape.join(",") !== y.shape.join(",")) {
      throw new Error("where requires all tensors to have the same shape.");
    }
    const length = product(c.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(WHERE_SHADER, "main");
    dispatchCompute(pipeline, [c.buffer, x.buffer, y.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, x.shape, x.dtype, length);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Float32Array([minVal, maxVal, length, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CLAMP_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async pow(aId: number, bId: number): Promise<TensorHandle> {
    if (aId !== bId) {
      return this.elementwise(aId, bId, "pow_op");
    }
    // Same tensor — pre-compute? just use elementwise anyway
    return this.elementwise(aId, bId, "pow_op");
  }

  async heaviside(inputId: number, valuesId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(inputId);
    const b = this.deviceMgr.getTensorMeta(valuesId);
    if (a.shape.join(",") !== b.shape.join(",")) {
      throw new Error("heaviside requires same shape for both inputs.");
    }
    const length = product(a.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, "heaviside");
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, length);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(aId);
    const b = this.deviceMgr.getTensorMeta(bId);

    // mv: (m, k) @ (k,) -> (m,)
    if (a.shape.length === 2 && b.shape.length === 1) {
      const [m, k] = a.shape;
      const [k2] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${m},${k}] x [${k2}].`);
      return this._runMatmul2d(m, k, 1, a.buffer, b.buffer, a.dtype);
    }

    // mv: (k,) @ (k, n) -> (n,)
    if (a.shape.length === 1 && b.shape.length === 2) {
      const [k] = a.shape;
      const [k2, n] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${k}] x [${k2},${n}].`);
      return this._runMatmul2d(1, k, n, a.buffer, b.buffer, b.dtype);
    }

    // bmm: (b, m, k) @ (b, k, n) -> (b, m, n)
    if (a.shape.length === 3 && b.shape.length === 3) {
      const [batch, m, k] = a.shape;
      const [batch2, k2, n] = b.shape;
      if (batch !== batch2) throw new Error(`matmul batch dim mismatch: ${batch} vs ${batch2}.`);
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${batch},${m},${k}] x [${batch2},${k2},${n}].`);
      return this._runMatmul3d(batch, m, k, n, a.buffer, b.buffer, a.dtype);
    }

    // mm: (m, k) @ (k, n) -> (m, n)
    if (a.shape.length === 2 && b.shape.length === 2) {
      const [m, k] = a.shape;
      const [k2, n] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [${m},${k}] x [${k2},${n}].`);
      return this._runMatmul2d(m, k, n, a.buffer, b.buffer, a.dtype);
    }

    // bmm broadcast: (..., m, k) @ (k, n) -> (..., m, n)
    if (a.shape.length >= 3 && b.shape.length === 2) {
      const batch = a.shape.slice(0, -2).reduce((x, y) => x * y, 1);
      const [m, k] = a.shape.slice(-2);
      const [k2, n] = b.shape;
      if (k !== k2) throw new Error(`matmul dimension mismatch: [...,${m},${k}] x [${k2},${n}].`);
      return this._runMatmul3d(batch, m, k, n, a.buffer, b.buffer, a.dtype);
    }

    throw new Error(`matmul unsupported shapes: [${a.shape}] and [${b.shape}].`);
  }

  private async _runMatmul2d(m: number, k: number, n: number,
    aBuf: GPUBuffer, bBuf: GPUBuffer, dtype: string): Promise<TensorHandle> {
    const out = createStorageBuffer(this.deviceMgr.device!, m * n * 4);
    const params = new Uint32Array([m, k, n, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul_2d");
    dispatchCompute(pipeline, [aBuf, bBuf, out, paramBuffer], calculateWorkgroups(m * n));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [m, n], dtype, m * n);
  }

  private async _runMatmul3d(batch: number, m: number, k: number, n: number,
    aBuf: GPUBuffer, bBuf: GPUBuffer, dtype: string): Promise<TensorHandle> {
    const out = createStorageBuffer(this.deviceMgr.device!, batch * m * n * 4);
    const params = new Uint32Array([m, k, n, batch]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul_3d");
    dispatchCompute(pipeline, [aBuf, bBuf, out, paramBuffer], calculateWorkgroups(batch * m * n));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [batch, m, n], dtype, batch * m * n);
  }

  private async elementwise(aId: number, bId: number, op: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const a = this.deviceMgr.getTensorMeta(aId);
    const b = this.deviceMgr.getTensorMeta(bId     );
    if (a.shape.join(",") !== b.shape.join(",")) {
      return this.broadcastOps.elementwiseWithBroadcast(a, b, op);
    }
    const length = product(a.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    return this.deviceMgr.registerTensorAsHandle(out, a.shape, a.dtype, length);
  }
}
