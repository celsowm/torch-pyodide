import { SupportedDType } from "./types.js";
import { DeviceManager } from "./device.js";
export { DeviceManager } from "./device.js";
export { sleep, withRetry } from "./device.js";

export {
  assertDType,
  coerceScalarByDType,
  decodeValuesByDType,
  assertUnaryDType,
  normalizeDim,
  computeStrides,
  linearToCoords,
  coordsToLinear,
  normalizeSliceStart,
  normalizeSliceEnd,
  padShapeTo4,
  broadcastShapes,
  getTensor,
} from "./shape.js";

import {
  BufferUsage,
  FILL_SHADER,
  RANDOM_SHADER,
  ELEMENTWISE_SHADER,
  MATMUL_SHADER,
  REDUCE_SUM_SHADER,
  CLAMP_SHADER,
  WHERE_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  UNARY_SHADER,
  TRANSPOSE_SHADER,
  CAT_SHADER,
  STACK_SHADER,
  PERMUTE_ND_SHADER,
  SELECT_SHADER,
  SLICE_SHADER,
  EXPAND_SHADER,
  INDEX_SELECT_SHADER,
  BROADCAST_SHADER,
  EXPAND_BROADCAST_SHADER,
  REDUCE_DIM_SHADER,
  COMPARE_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  MASKED_SELECT_SHADER,
  MASKED_FILL_SHADER,
  LEAKY_RELU_SHADER,
  TRIL_SHADER,
  TRIU_SHADER,
  FLIP_SHADER,
  LOG_SOFTMAX_SHADER,
  REPEAT_SHADER,
  GATHER_SHADER,
  SORT_SHADER,
  CHOLESKY_SHADER,
  LU_SHADER,
  TRIANGULAR_SOLVE_SHADER,
  CONV_SHADER,
  MAX_POOL2D_SHADER,
  AVG_POOL2D_SHADER,
  BATCHNORM_SHADER,
  LAYERNORM_SHADER,
  NLL_LOSS_SHADER,
  CONV_BACKWARD_SHADER,
  SLICE_BACKWARD_SHADER,
  LOG_SOFTMAX_BACKWARD_SHADER,
  SOFTMAX_BACKWARD_SHADER,
  NLL_LOSS_BACKWARD_SHADER,
  REDUCE_BROADCAST_GRAD_SHADER,
  SORT_BACKWARD_SHADER,
  TOPK_BACKWARD_SHADER,
  CROSS_ENTROPY_SHADER,
  CROSS_ENTROPY_BACKWARD_SHADER,
  ADAM_STEP_SHADER,
  SOFTMAX_SHADER,
  ADAMW_STEP_SHADER,
  NLL_LOSS_REDUCED_SHADER,
  SGD_STEP_SHADER,
  RMSPROP_STEP_SHADER,
  MAXMIN_BACKWARD_SHADER,
  TERNARY_SHADER,
  EXTENDED_STEP_SHADER,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
} from "../vendor/torchjs/index.js";

export {
  BufferUsage,
  FILL_SHADER,
  RANDOM_SHADER,
  ELEMENTWISE_SHADER,
  MATMUL_SHADER,
  REDUCE_SUM_SHADER,
  CLAMP_SHADER,
  WHERE_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  UNARY_SHADER,
  TRANSPOSE_SHADER,
  CAT_SHADER,
  STACK_SHADER,
  PERMUTE_ND_SHADER,
  SELECT_SHADER,
  SLICE_SHADER,
  EXPAND_SHADER,
  INDEX_SELECT_SHADER,
  BROADCAST_SHADER,
  EXPAND_BROADCAST_SHADER,
  REDUCE_DIM_SHADER,
  COMPARE_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  MASKED_SELECT_SHADER,
  MASKED_FILL_SHADER,
  LEAKY_RELU_SHADER,
  TRIL_SHADER,
  TRIU_SHADER,
  FLIP_SHADER,
  LOG_SOFTMAX_SHADER,
  REPEAT_SHADER,
  GATHER_SHADER,
  SORT_SHADER,
  CHOLESKY_SHADER,
  LU_SHADER,
  TRIANGULAR_SOLVE_SHADER,
  CONV_SHADER,
  MAX_POOL2D_SHADER,
  AVG_POOL2D_SHADER,
  BATCHNORM_SHADER,
  LAYERNORM_SHADER,
  NLL_LOSS_SHADER,
  CONV_BACKWARD_SHADER,
  SLICE_BACKWARD_SHADER,
  LOG_SOFTMAX_BACKWARD_SHADER,
  SOFTMAX_BACKWARD_SHADER,
  NLL_LOSS_BACKWARD_SHADER,
  REDUCE_BROADCAST_GRAD_SHADER,
  SORT_BACKWARD_SHADER,
  TOPK_BACKWARD_SHADER,
  CROSS_ENTROPY_SHADER,
  CROSS_ENTROPY_BACKWARD_SHADER,
  ADAM_STEP_SHADER,
  SOFTMAX_SHADER,
  ADAMW_STEP_SHADER,
  NLL_LOSS_REDUCED_SHADER,
  SGD_STEP_SHADER,
  RMSPROP_STEP_SHADER,
  MAXMIN_BACKWARD_SHADER,
  TERNARY_SHADER,
  EXTENDED_STEP_SHADER,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  makeStorageReadLayoutEntry,
  makeStorageReadWriteLayoutEntry,
  makeUniformLayoutEntry,
} from "../vendor/torchjs/index.js";

let _deviceMgr: DeviceManager | null = null;

export function setDeviceManager(mgr: DeviceManager): void {
  _deviceMgr = mgr;
}

export function createStorageBuffer(_device: GPUDevice, size: number): GPUBuffer {
  if (!_deviceMgr) throw new Error("DeviceManager not set");
  return _deviceMgr.createStorageBuffer(size);
}

export function getShadowId(_buffer: GPUBuffer): number | undefined {
  return undefined;
}

export async function readScalar(_device: GPUDevice, buffer: GPUBuffer): Promise<number> {
  if (!_deviceMgr) throw new Error("DeviceManager not set");
  return _deviceMgr.readScalar(buffer);
}

export async function readFromGPU(_device: GPUDevice, source: GPUBuffer, length: number, dtype: SupportedDType): Promise<number[]> {
  if (!_deviceMgr) throw new Error("DeviceManager not set");
  return _deviceMgr.readFromGPU(source, length, dtype);
}
