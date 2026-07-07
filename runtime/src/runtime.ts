import { TensorHandle, SupportedDType } from "./ops/types.js";
import { setDeviceManager } from "./ops/utils.js";
import { DeviceManager } from "./ops/device.js";
import { CreationOps } from "./ops/creationOps.js";
import { ArithmeticOps } from "./ops/arithmeticOps.js";
import { UnaryOps } from "./ops/unaryOps.js";
import { ReductionOps } from "./ops/reductionOps.js";
import { ShapeOps } from "./ops/shapeOps.js";
import { CompareOps } from "./ops/compareOps.js";
import { MaskingOps } from "./ops/maskingOps.js";
import { LinalgOps } from "./ops/linalgOps.js";
import { ConvOps } from "./ops/convOps.js";
import { PoolingOps } from "./ops/poolingOps.js";
import { NormalizationOps } from "./ops/normalizationOps.js";
import { EmbeddingOps } from "./ops/embeddingOps.js";

export class TorchPyodideRuntime {
  private deviceMgr = new DeviceManager();
  private creationOps: CreationOps;
  private arithmeticOps: ArithmeticOps;
  private unaryOps: UnaryOps;
  private reductionOps: ReductionOps;
  private shapeOps: ShapeOps;
  private compareOps: CompareOps;
  private maskingOps: MaskingOps;
  private linalgOps: LinalgOps;
  private convOps: ConvOps;
  private poolingOps: PoolingOps;
  private normalizationOps: NormalizationOps;
  private embeddingOps: EmbeddingOps;

  constructor() {
    setDeviceManager(this.deviceMgr);
    const dm = this.deviceMgr;
    this.creationOps = new CreationOps(dm);
    this.arithmeticOps = new ArithmeticOps(dm);
    this.unaryOps = new UnaryOps(dm);
    this.reductionOps = new ReductionOps(dm);
    this.shapeOps = new ShapeOps(dm);
    this.compareOps = new CompareOps(dm);
    this.maskingOps = new MaskingOps(dm);
    this.linalgOps = new LinalgOps(dm);
    this.convOps = new ConvOps(dm);
    this.poolingOps = new PoolingOps(dm);
    this.normalizationOps = new NormalizationOps(dm);
    this.embeddingOps = new EmbeddingOps(dm);
  }

  async init(gpuProvider?: GPU | null): Promise<void> {
    await this.deviceMgr.ensureReady(gpuProvider);
  }

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.tensorFromData(data, shape, dtype);
  }

  async zeros(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.zeros(shape, dtype);
  }

  async ones(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.ones(shape, dtype);
  }

  async setSeed(seed: number): Promise<void> {
    return this.creationOps.setSeed(seed);
  }

  async rand(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.rand(shape, dtype);
  }

  async randn(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.randn(shape, dtype);
  }

  async normal(shape: number[], dtype: string, mean: number, std: number): Promise<TensorHandle> {
    return this.creationOps.normal(shape, dtype, mean, std);
  }

  async bernoulli(shape: number[], dtype: string, p: number): Promise<TensorHandle> {
    return this.creationOps.bernoulli(shape, dtype, p);
  }

  async exponential(shape: number[], dtype: string, rate: number): Promise<TensorHandle> {
    return this.creationOps.exponential(shape, dtype, rate);
  }

  async logNormal(shape: number[], dtype: string, mean: number, std: number): Promise<TensorHandle> {
    return this.creationOps.logNormal(shape, dtype, mean, std);
  }

  async arange(start: number, end: number, step: number, dtype: string): Promise<TensorHandle> {
    return this.creationOps.arange(start, end, step, dtype);
  }

  async full(shape: number[], fillValue: number, dtype: string): Promise<TensorHandle> {
    return this.creationOps.full(shape, fillValue, dtype);
  }

  async fullLike(tensorId: number, fillValue: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.fullLike(tensorId, fillValue, dtype);
  }

  async zerosLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.zerosLike(tensorId, dtype);
  }

  async onesLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.onesLike(tensorId, dtype);
  }

  async emptyLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.emptyLike(tensorId, dtype);
  }

  async empty(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.empty(shape, dtype);
  }

  async add(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.add(aId, bId);
  }

  async mul(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.mul(aId, bId);
  }

  async sub(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.sub(aId, bId);
  }

  async div(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.div(aId, bId);
  }

  async where(conditionId: number, xId: number, yId: number): Promise<TensorHandle> {
    return this.arithmeticOps.where(conditionId, xId, yId);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    return this.arithmeticOps.clamp(tensorId, minVal, maxVal);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.matmul(aId, bId);
  }

  async mm(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.matmul(aId, bId);
  }

  async bmm(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.matmul(aId, bId);
  }

  async mv(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.matmul(aId, bId);
  }

  async pow(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.pow(aId, bId);
  }

  async heaviside(inputId: number, valuesId: number): Promise<TensorHandle> {
    return this.arithmeticOps.heaviside(inputId, valuesId);
  }

  async atan2(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.atan2(aId, bId);
  }

  async hypot(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.hypot(aId, bId);
  }

  async logaddexp(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.logaddexp(aId, bId);
  }

  async logaddexp2(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.logaddexp2(aId, bId);
  }

  async fmod(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.fmod(aId, bId);
  }

  async remainder(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.remainder(aId, bId);
  }

  async xlogy(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.xlogy(aId, bId);
  }

  async copysign(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.copysign(aId, bId);
  }

  async floorDivide(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.floorDivide(aId, bId);
  }

  async trueDivide(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.trueDivide(aId, bId);
  }

  async nextafter(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.nextafter(aId, bId);
  }

  async logicalAnd(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.logicalAnd(aId, bId);
  }

  async logicalOr(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.logicalOr(aId, bId);
  }

  async logicalXor(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.logicalXor(aId, bId);
  }

  async bitwiseAnd(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.bitwiseAnd(aId, bId);
  }

  async bitwiseOr(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.bitwiseOr(aId, bId);
  }

  async bitwiseXor(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.bitwiseXor(aId, bId);
  }

  async bitwiseNot(aId: number): Promise<TensorHandle> {
    return this.arithmeticOps.bitwiseNot(aId);
  }

  async lerpScalar(startId: number, endId: number, weight: number): Promise<TensorHandle> {
    return this.arithmeticOps.lerpScalar(startId, endId, weight);
  }

  async lerpTensor(startId: number, endId: number, weightId: number): Promise<TensorHandle> {
    return this.arithmeticOps.lerpTensor(startId, endId, weightId);
  }

  async addcmul(inputId: number, t1Id: number, t2Id: number, value: number): Promise<TensorHandle> {
    return this.arithmeticOps.addcmul(inputId, t1Id, t2Id, value);
  }

  async addcdiv(inputId: number, t1Id: number, t2Id: number, value: number): Promise<TensorHandle> {
    return this.arithmeticOps.addcdiv(inputId, t1Id, t2Id, value);
  }

  async mulScalar(tensorId: number, value: number): Promise<TensorHandle> {
    return this.arithmeticOps.mulScalar(tensorId, value);
  }

  async relu(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.relu(tensorId);
  }

  async abs(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.abs(tensorId);
  }

  async sqrt(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sqrt(tensorId);
  }

  async exp(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.exp(tensorId);
  }

  async log(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.log(tensorId);
  }

  async neg(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.neg(tensorId);
  }

  async sigmoid(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sigmoid(tensorId);
  }

  async tanh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.tanh(tensorId);
  }

  async sin(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sin(tensorId);
  }

  async cos(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.cos(tensorId);
  }

  async gelu(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.gelu(tensorId);
  }

  async silu(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.silu(tensorId);
  }

  async leakyRelu(tensorId: number, alpha = 0.01): Promise<TensorHandle> {
    return this.unaryOps.leakyRelu(tensorId, alpha);
  }

  async floor(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.floor(tensorId);
  }

  async ceil(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.ceil(tensorId);
  }

  async round(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.round(tensorId);
  }

  async reciprocal(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.reciprocal(tensorId);
  }

  async square(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.square(tensorId);
  }

  // Trig
  async tan(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.tan(tensorId);
  }

  async asin(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.asin(tensorId);
  }

  async acos(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.acos(tensorId);
  }

  async atan(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.atan(tensorId);
  }

  async sinh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sinh(tensorId);
  }

  async cosh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.cosh(tensorId);
  }

  async asinh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.asinh(tensorId);
  }

  async acosh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.acosh(tensorId);
  }

  async atanh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.atanh(tensorId);
  }

  // Exp/Log
  async exp2(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.exp2(tensorId);
  }

  async log2(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.log2(tensorId);
  }

  async log10(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.log10(tensorId);
  }

  async log1p(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.log1p(tensorId);
  }

  async expm1(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.expm1(tensorId);
  }

  // Rounding
  async trunc(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.trunc(tensorId);
  }

  async frac(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.frac(tensorId);
  }

  // Activations
  async softplus(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.softplus(tensorId);
  }

  async mish(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.mish(tensorId);
  }

  async hardsigmoid(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.hardsigmoid(tensorId);
  }

  async hardswish(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.hardswish(tensorId);
  }

  async softsign(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.softsign(tensorId);
  }

  async tanhshrink(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.tanhshrink(tensorId);
  }

  // Arithmetic
  async rsqrt(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.rsqrt(tensorId);
  }

  async sign(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sign(tensorId);
  }

  async sgn(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sgn(tensorId);
  }

  // Boolean
  async isnan(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.isnan(tensorId);
  }

  async isinf(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.isinf(tensorId);
  }

  async isfinite(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.isfinite(tensorId);
  }

  async isposinf(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.isposinf(tensorId);
  }

  async isneginf(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.isneginf(tensorId);
  }

  async logicalNot(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.logicalNot(tensorId);
  }

  // Special
  async erf(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.erf(tensorId);
  }

  async erfc(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.erfc(tensorId);
  }

  async lgamma(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.lgamma(tensorId);
  }

  async digamma(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.digamma(tensorId);
  }

  async i0(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.i0(tensorId);
  }

  // Conversion
  async deg2rad(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.deg2rad(tensorId);
  }

  async rad2deg(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.rad2deg(tensorId);
  }

  async fill(tensorId: number, value: number): Promise<TensorHandle> {
    return this.unaryOps.fill(tensorId, value);
  }

  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.sum(tensorId);
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.mean(tensorId);
  }

  async sumDim(tensorId: number, dim: number, keepdim: boolean): Promise<TensorHandle> {
    return this.reductionOps.sumDim(tensorId, dim, keepdim);
  }

  async meanDim(tensorId: number, dim: number, keepdim: boolean): Promise<TensorHandle> {
    return this.reductionOps.meanDim(tensorId, dim, keepdim);
  }

  async prod(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.prod(tensorId);
  }

  async min(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.min(tensorId);
  }

  async max(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.max(tensorId);
  }

  async argmax(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.argmax(tensorId);
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.argmin(tensorId);
  }

  async any(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.any(tensorId);
  }

  async all(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.all(tensorId);
  }

  async cumsum(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.cumsum(tensorId);
  }

  async cumprod(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.cumprod(tensorId);
  }

  async softmax(tensorId: number, dim: number): Promise<TensorHandle> {
    return this.reductionOps.softmax(tensorId, dim);
  }

  async logSoftmax(tensorId: number, dim: number): Promise<TensorHandle> {
    return this.reductionOps.logSoftmax(tensorId, dim);
  }

  async nllLoss(inputId: number, targetsId: number): Promise<TensorHandle> {
    return this.reductionOps.nllLoss(inputId, targetsId);
  }

  async nllLossReduced(inputId: number, targetsId: number, reduction: "sum" | "mean"): Promise<TensorHandle> {
    return this.reductionOps.nllLossReduced(inputId, targetsId, reduction);
  }

  async crossEntropy(inputId: number, targetsId: number): Promise<TensorHandle> {
    return this.reductionOps.crossEntropy(inputId, targetsId);
  }

  async logSoftmaxBackward(
    gradOutputId: number,
    softmaxId: number,
    batchSize: number,
    numClasses: number,
  ): Promise<TensorHandle> {
    return this.reductionOps.logSoftmaxBackward(gradOutputId, softmaxId, batchSize, numClasses);
  }

  async softmaxBackward(
    gradOutputId: number,
    softmaxId: number,
    batchSize: number,
    numClasses: number,
  ): Promise<TensorHandle> {
    return this.reductionOps.softmaxBackward(gradOutputId, softmaxId, batchSize, numClasses);
  }

  async nllLossBackward(
    targetsId: number,
    batchSize: number,
    numClasses: number,
    scale?: number,
  ): Promise<TensorHandle> {
    return this.reductionOps.nllLossBackward(targetsId, batchSize, numClasses, scale);
  }

  async crossEntropyBackward(
    gradOutputId: number,
    inputId: number,
    targetsId: number,
    reduction: "none" | "sum" | "mean",
  ): Promise<TensorHandle> {
    return this.reductionOps.crossEntropyBackward(gradOutputId, inputId, targetsId, reduction);
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
    return this.reductionOps.adamStep(
      paramId,
      gradId,
      expAvgId,
      expAvgSqId,
      lr,
      beta1,
      beta2,
      eps,
      weightDecay,
      stepSize,
      invSqrtBiasCorrection2,
    );
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
    return this.reductionOps.adamWStep(
      paramId,
      gradId,
      expAvgId,
      expAvgSqId,
      lr,
      beta1,
      beta2,
      eps,
      weightDecay,
      stepSize,
      invSqrtBiasCorrection2,
    );
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
    return this.reductionOps.sgdStep(
      paramId,
      gradId,
      momentumBufId,
      lr,
      momentum,
      weightDecay,
      dampening,
      nesterov,
    );
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
    return this.reductionOps.rmspropStep(
      paramId,
      gradId,
      squareAvgId,
      momentumBufId,
      lr,
      alpha,
      eps,
      weightDecay,
      momentum,
    );
  }

  async adagradStep(
    paramId: number,
    gradId: number,
    sumSquaresId: number,
    lr: number,
    eps: number,
    weightDecay: number,
  ): Promise<void> {
    return this.reductionOps.adagradStep(paramId, gradId, sumSquaresId, lr, eps, weightDecay);
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
    return this.reductionOps.adamaxStep(
      paramId,
      gradId,
      expAvgId,
      expInfId,
      lr,
      beta1,
      beta2,
      eps,
      weightDecay,
      stepSize,
      biasCorrection1,
    );
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
    return this.reductionOps.nadamStep(
      paramId,
      gradId,
      expAvgId,
      expAvgSqId,
      lr,
      beta1,
      beta2,
      eps,
      weightDecay,
      stepSize,
      mu,
    );
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
  ): Promise<void> {
    return this.reductionOps.radamStep(
      paramId,
      gradId,
      expAvgId,
      expAvgSqId,
      lr,
      beta1,
      beta2,
      eps,
      weightDecay,
      stepSize,
      beta1PowT,
      beta2PowT,
    );
  }

  async maxMinBackward(inputId: number, gradOutputId: number, mode: "max" | "min"): Promise<TensorHandle> {
    return this.reductionOps.maxMinBackward(inputId, gradOutputId, mode);
  }

  async eq(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.eq(aId, bId);
  }

  async ne(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.ne(aId, bId);
  }

  async lt(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.lt(aId, bId);
  }

  async le(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.le(aId, bId);
  }

  async gt(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.gt(aId, bId);
  }

  async ge(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.ge(aId, bId);
  }

  async maximum(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.maximum(aId, bId);
  }

  async minimum(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.minimum(aId, bId);
  }

  async maskedSelect(tensorId: number, maskId: number): Promise<TensorHandle> {
    return this.maskingOps.maskedSelect(tensorId, maskId);
  }

  async maskedFill(tensorId: number, maskId: number, value: number): Promise<TensorHandle> {
    return this.maskingOps.maskedFill(tensorId, maskId, value);
  }

  async reshape(tensorId: number, shape: number[]): Promise<TensorHandle> {
    return this.shapeOps.reshape(tensorId, shape);
  }

  async flatten(tensorId: number, startDim = 0, endDim = -1): Promise<TensorHandle> {
    return this.shapeOps.flatten(tensorId, startDim, endDim);
  }

  async squeeze(tensorId: number, dim?: number): Promise<TensorHandle> {
    return this.shapeOps.squeeze(tensorId, dim);
  }

  async unsqueeze(tensorId: number, dim: number): Promise<TensorHandle> {
    return this.shapeOps.unsqueeze(tensorId, dim);
  }

  async transpose2d(tensorId: number): Promise<TensorHandle> {
    return this.shapeOps.transpose2d(tensorId);
  }

  async transpose(tensorId: number, dim0: number, dim1: number): Promise<TensorHandle> {
    return this.shapeOps.transpose(tensorId, dim0, dim1);
  }

  async permute(tensorId: number, dims: number[]): Promise<TensorHandle> {
    return this.shapeOps.permute(tensorId, dims);
  }

  async select(tensorId: number, dim: number, index: number): Promise<TensorHandle> {
    return this.shapeOps.select(tensorId, dim, index);
  }

  async slice(tensorId: number, dim: number, start?: number, end?: number, step = 1): Promise<TensorHandle> {
    return this.shapeOps.slice(tensorId, dim, start, end, step);
  }

  async sliceBackward(
    gradOutputId: number,
    inputShape: number[],
    slicedShape: number[],
    dim: number,
    start: number,
    step: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.sliceBackward(gradOutputId, inputShape, slicedShape, dim, start, step);
  }

  async cat(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.shapeOps.cat(tensorIds, dim);
  }

  async stack(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.shapeOps.stack(tensorIds, dim);
  }

  async expand(tensorId: number, shape: number[]): Promise<TensorHandle> {
    return this.shapeOps.expand(tensorId, shape);
  }

  async indexSelect(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    return this.shapeOps.indexSelect(tensorId, dim, indicesId);
  }

  async gather(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    return this.shapeOps.gather(tensorId, dim, indicesId);
  }

  async scatter(tensorId: number, dim: number, indexId: number, srcId: number): Promise<TensorHandle> {
    return this.shapeOps.scatter(tensorId, dim, indexId, srcId);
  }

  async scatterAdd(tensorId: number, dim: number, indexId: number, srcId: number): Promise<TensorHandle> {
    return this.shapeOps.scatterAdd(tensorId, dim, indexId, srcId);
  }

  async sort(tensorId: number, dim: number): Promise<TensorHandle[]> {
    return this.shapeOps.sort(tensorId, dim);
  }

  async sortBackward(
    gradOutputId: number,
    indicesId: number,
    inputShape: number[],
    dim: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.sortBackward(gradOutputId, indicesId, inputShape, dim);
  }

  async topkBackward(
    gradOutputId: number,
    indicesId: number,
    inputShape: number[],
    dim: number,
    k: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.topkBackward(gradOutputId, indicesId, inputShape, dim, k);
  }

  async tril(tensorId: number, diagonal = 0): Promise<TensorHandle> {
    return this.shapeOps.tril(tensorId, diagonal);
  }

  async triu(tensorId: number, diagonal = 0): Promise<TensorHandle> {
    return this.shapeOps.triu(tensorId, diagonal);
  }

  async flip(tensorId: number, dims: number[]): Promise<TensorHandle> {
    return this.shapeOps.flip(tensorId, dims);
  }

  async nonzero(tensorId: number): Promise<{ count: number; indices: TensorHandle }> {
    return this.shapeOps.nonzero(tensorId);
  }

  async roll(tensorId: number, shift: number): Promise<TensorHandle> {
    return this.shapeOps.roll(tensorId, shift);
  }

  async replicationPad(
    tensorId: number,
    padLeft: number,
    padRight: number,
    padTop: number,
    padBottom: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.replicationPad(tensorId, padLeft, padRight, padTop, padBottom);
  }

  async reflectionPad(
    tensorId: number,
    padLeft: number,
    padRight: number,
    padTop: number,
    padBottom: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.reflectionPad(tensorId, padLeft, padRight, padTop, padBottom);
  }

  async circularPad(
    tensorId: number,
    padLeft: number,
    padRight: number,
    padTop: number,
    padBottom: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.circularPad(tensorId, padLeft, padRight, padTop, padBottom);
  }

  async constantPad(
    tensorId: number,
    padLeft: number,
    padRight: number,
    padTop: number,
    padBottom: number,
    value: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.constantPad(tensorId, padLeft, padRight, padTop, padBottom, value);
  }

  async upsample2d(
    tensorId: number,
    outH: number,
    outW: number,
    mode: number,
    alignCorners: number,
  ): Promise<TensorHandle> {
    return this.shapeOps.upsample2d(tensorId, outH, outW, mode, alignCorners);
  }

  async repeat(tensorId: number, sizes: number[]): Promise<TensorHandle> {
    return this.shapeOps.repeat(tensorId, sizes);
  }

  /** Execute a batch of operations — all compute work is accumulated and submitted once. */
  async runBatch<T>(fn: () => Promise<T>): Promise<T> {
    this.deviceMgr.beginFrame();
    try {
      const result = await fn();
      await this.deviceMgr.endFrame();
      return result;
    } catch (err) {
      this.deviceMgr.cancelFrame();
      throw err;
    }
  }

  beginFrame(): void {
    this.deviceMgr.beginFrame();
  }

  endFrame(): Promise<void> {
    return this.deviceMgr.endFrame();
  }

  cancelFrame(): void {
    this.deviceMgr.cancelFrame();
  }

  async toList(tensorId: number): Promise<number[]> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    return this.deviceMgr.readFromGPU(meta.buffer, meta.length, meta.dtype as SupportedDType);
  }

  async destroy(tensorId: number): Promise<void> {
    this.deviceMgr.destroyTensor(tensorId);
  }

  isAvailable(): boolean {
    return this.deviceMgr.isAvailable();
  }

  isInitialized(): boolean {
    return this.deviceMgr.initialized;
  }

  deviceCount(): number {
    return this.deviceMgr.deviceCount();
  }

  async currentDevice(): Promise<number> {
    return this.deviceMgr.currentDevice();
  }

  async getDeviceName(deviceIndex?: number): Promise<string> {
    return this.deviceMgr.getDeviceName(deviceIndex);
  }

  async getDeviceProperties(deviceIndex?: number): Promise<Record<string, unknown>> {
    return this.deviceMgr.getDeviceProperties(deviceIndex);
  }

  async memoryAllocated(_deviceIndex?: number): Promise<number> {
    return this.deviceMgr.memoryAllocated();
  }

  async memoryReserved(_deviceIndex?: number): Promise<number> {
    return this.deviceMgr.memoryReserved();
  }

  async conv2d(
    inputId: number,
    weightId: number,
    bias: number[] | null,
    stride: number[],
    padding: number[],
    dilation: number[],
    groups: number,
  ): Promise<TensorHandle> {
    return this.convOps.conv2d(inputId, weightId, bias, stride, padding, dilation, groups);
  }

  async conv2dInputBackward(
    gradOutputId: number,
    weightId: number,
    inputShape: number[],
    gradOutputShape: number[],
    stride: number[],
    padding: number[],
  ): Promise<TensorHandle> {
    return this.convOps.conv2dInputBackward(gradOutputId, weightId, inputShape, gradOutputShape, stride, padding);
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
    return this.convOps.conv2dWeightBackward(gradOutputId, inputId, weightShape, gradOutputShape, inputShape, stride, padding);
  }

  async conv2dBiasBackward(
    gradOutputId: number,
    outCh: number,
    gradOutputShape: number[],
  ): Promise<TensorHandle> {
    return this.convOps.conv2dBiasBackward(gradOutputId, outCh, gradOutputShape);
  }

  async maxPool2d(
    inputId: number,
    kernelSize: number[],
    stride: number[],
    padding: number[],
    dilation: number[],
  ): Promise<TensorHandle> {
    return this.poolingOps.maxPool2d(inputId, kernelSize, stride, padding, dilation);
  }

  async avgPool2d(
    inputId: number,
    kernelSize: number[],
    stride: number[],
    padding: number[],
    countIncludePad: boolean,
  ): Promise<TensorHandle> {
    return this.poolingOps.avgPool2d(inputId, kernelSize, stride, padding, countIncludePad);
  }

  async batchNorm(
    inputId: number,
    weightId: number | null,
    biasId: number | null,
    runningMeanId: number | null,
    runningVarId: number | null,
    eps: number,
  ): Promise<TensorHandle> {
    return this.normalizationOps.batchNorm(inputId, weightId, biasId, runningMeanId, runningVarId, eps);
  }

  async layerNorm(
    inputId: number,
    normalizedShape: number[],
    gammaId: number | null,
    betaId: number | null,
    eps: number,
  ): Promise<TensorHandle> {
    return this.normalizationOps.layerNorm(inputId, normalizedShape, gammaId, betaId, eps);
  }

  async embedding(
    weightId: number,
    indicesId: number,
    numEmbeddings: number,
    embeddingDim: number,
    paddingIdx: number,
  ): Promise<TensorHandle> {
    return this.embeddingOps.embedding(weightId, indicesId, numEmbeddings, embeddingDim, paddingIdx);
  }

  async cholesky(tensorId: number): Promise<TensorHandle> {
    return this.linalgOps.cholesky(tensorId);
  }

  async lu(tensorId: number): Promise<TensorHandle[]> {
    return this.linalgOps.lu(tensorId);
  }

  async triangularSolve(aId: number, bId: number, upper: boolean): Promise<TensorHandle> {
    return this.linalgOps.triangularSolve(aId, bId, upper);
  }
}

export function installTorchRuntime(target: typeof globalThis = globalThis): TorchPyodideRuntime {
  const runtime = new TorchPyodideRuntime();
  (target as typeof globalThis & { __torch_pyodide_runtime__?: TorchPyodideRuntime }).__torch_pyodide_runtime__ =
    runtime;
  return runtime;
}
