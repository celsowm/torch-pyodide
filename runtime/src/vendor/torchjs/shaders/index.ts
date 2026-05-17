/**
 * Shader code exports.
 * WGSL shaders are loaded from .wgsl files and bundled during build.
 * @status implemented
 */

import ELEMENTWISE_SHADER from './elementwise.wgsl';
import SCALAR_SHADER from './scalar.wgsl';
import UNARY_SHADER from './unary.wgsl';
import FILL_SHADER from './fill.wgsl';
import RANDOM_SHADER from './random.wgsl';
import REDUCE_SUM_SHADER from './reduce_sum.wgsl';
import REDUCE_MAX_SHADER from './reduce_max.wgsl';
import REDUCE_MIN_SHADER from './reduce_min.wgsl';
import REDUCE_ANY_SHADER from './reduce_any.wgsl';
import REDUCE_ALL_SHADER from './reduce_all.wgsl';
import REDUCE_PROD_SHADER from './reduce_prod.wgsl';
import REDUCE_SIMPLE_SUM_SHADER from './reduce_simple_sum.wgsl';
import REDUCE_SIMPLE_MAX_SHADER from './reduce_simple_max.wgsl';
import REDUCE_SIMPLE_MIN_SHADER from './reduce_simple_min.wgsl';
import REDUCE_SIMPLE_ANY_SHADER from './reduce_simple_any.wgsl';
import REDUCE_SIMPLE_ALL_SHADER from './reduce_simple_all.wgsl';
import REDUCE_SIMPLE_PROD_SHADER from './reduce_simple_prod.wgsl';
import REDUCE_DIM_SHADER from './reduce_dim.wgsl';
import MATMUL_SHADER from './matmul.wgsl';
import TRANSPOSE_SHADER from './transpose.wgsl';
import LOG_SOFTMAX_SHADER from './log_softmax.wgsl';
import NLL_LOSS_SHADER from './nll_loss.wgsl';
import ARGMAX_SHADER from './argmax.wgsl';
import ARGMIN_SHADER from './argmin.wgsl';
import COMPARE_SHADER from './compare.wgsl';
import INDEX_SELECT_SHADER from './index_select.wgsl';
import EXPAND_SHADER from './expand.wgsl';
import SLICE_SHADER from './slice.wgsl';
import SLICE_BACKWARD_SHADER from './slice_backward.wgsl';
import MASKED_SELECT_SHADER from './masked_select.wgsl';
import BROADCAST_SHADER from './broadcast.wgsl';
import NLL_LOSS_BACKWARD_SHADER from './nll_loss_backward.wgsl';
import LOG_SOFTMAX_BACKWARD_SHADER from './log_softmax_backward.wgsl';
import MASKED_FILL_SHADER from './masked_fill.wgsl';
import WHERE_SHADER from './where.wgsl';
import TRANSPOSE_ND_SHADER from './transpose_nd.wgsl';
import REDUCE_BROADCAST_GRAD_SHADER from './reduce_broadcast_grad.wgsl';
import EMBEDDING_SHADER from './embedding.wgsl';
import LAYERNORM_SHADER from './layernorm.wgsl';
import TRIL_SHADER from './tril.wgsl';
import TRIU_SHADER from './triu.wgsl';
import FLIP_SHADER from './flip.wgsl';
import HEAVISIDE_SHADER from './heaviside.wgsl';
import CUMSUM_SHADER from './cumsum.wgsl';
import CUMPROD_SHADER from './cumprod.wgsl';
import DIAG_VEC_TO_MTX_SHADER from './diag_vec_to_mtx.wgsl';
import DIAG_MTX_TO_VEC_SHADER from './diag_mtx_to_vec.wgsl';
import CLAMP_SHADER from './clamp.wgsl';
import CHOLESKY_SHADER from './cholesky.wgsl';
import TRIANGULAR_SOLVE_SHADER from './triangular_solve.wgsl';
import LU_SHADER from './lu.wgsl';
import CONV_SHADER from './conv.wgsl';
import CONV_BACKWARD_SHADER from './conv_backward.wgsl';
import MAX_POOL2D_SHADER from './max_pool2d.wgsl';
import AVG_POOL2D_SHADER from './avg_pool2d.wgsl';
import BATCHNORM_SHADER from './batchnorm.wgsl';
import PERMUTE_SHADER from './permute.wgsl';

export {
  ELEMENTWISE_SHADER,
  SCALAR_SHADER,
  UNARY_SHADER,
  FILL_SHADER,
  RANDOM_SHADER,
  REDUCE_SUM_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  REDUCE_ANY_SHADER,
  REDUCE_ALL_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_SIMPLE_SUM_SHADER,
  REDUCE_SIMPLE_MAX_SHADER,
  REDUCE_SIMPLE_MIN_SHADER,
  REDUCE_SIMPLE_ANY_SHADER,
  REDUCE_SIMPLE_ALL_SHADER,
  REDUCE_SIMPLE_PROD_SHADER,
  REDUCE_DIM_SHADER,
  MATMUL_SHADER,
  TRANSPOSE_SHADER,
  LOG_SOFTMAX_SHADER,
  NLL_LOSS_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  COMPARE_SHADER,
  INDEX_SELECT_SHADER,
  EXPAND_SHADER,
  SLICE_SHADER,
  SLICE_BACKWARD_SHADER,
  MASKED_SELECT_SHADER,
  BROADCAST_SHADER,
  NLL_LOSS_BACKWARD_SHADER,
  LOG_SOFTMAX_BACKWARD_SHADER,
  MASKED_FILL_SHADER,
  WHERE_SHADER,
  TRANSPOSE_ND_SHADER,
  REDUCE_BROADCAST_GRAD_SHADER,
  EMBEDDING_SHADER,
  LAYERNORM_SHADER,
  TRIL_SHADER,
  TRIU_SHADER,
  FLIP_SHADER,
  HEAVISIDE_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  DIAG_VEC_TO_MTX_SHADER,
  DIAG_MTX_TO_VEC_SHADER,
  CLAMP_SHADER,
  CHOLESKY_SHADER,
  TRIANGULAR_SOLVE_SHADER,
  LU_SHADER,
  CONV_SHADER,
  CONV_BACKWARD_SHADER,
  MAX_POOL2D_SHADER,
  AVG_POOL2D_SHADER,
  BATCHNORM_SHADER,
  PERMUTE_SHADER,
};
