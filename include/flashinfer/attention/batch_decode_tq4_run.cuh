/*
 * Batch decode with TQ4 paged KV cache + automatic partition-kv planning.
 *
 * Mirrors batch_decode_run.cuh but for TQ4-quantized KV cache.
 */

#ifndef FLASHINFER_ATTENTION_BATCH_DECODE_TQ4_RUN_CUH_
#define FLASHINFER_ATTENTION_BATCH_DECODE_TQ4_RUN_CUH_

#include <algorithm>
#include <vector>

#include <flashinfer/attention/decode_tq4.cuh>
#include <flashinfer/page_tq4.cuh>

namespace flashinfer {

/*!
 * \brief Batch decode with TQ4 paged KV cache, automatic partition-kv.
 * \tparam HEAD_DIM Head dimension (compile-time constant)
 * \tparam DTypeQ Query dtype (e.g. nv_bfloat16)
 * \tparam DTypeO Output dtype
 * \tparam IdType Index type (e.g. int32_t)
 * \tparam AttentionVariant Attention variant
 */
template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeO,
          typename IdType, typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheTQ4Run(
    DTypeQ* q,
    paged_kv_tq4_t<IdType> paged_kv,
    DTypeO* o,
    const IdType* h_kv_lens,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    float sm_scale,
    cudaStream_t stream,
    uint32_t kv_chunk_threshold = 256,
    uint32_t kv_chunk_size = 256) {

  using TQ4Params = BatchDecodeParamsTQ4<DTypeQ, DTypeO, IdType>;

  IdType max_kv_len = 0;
  for (uint32_t i = 0; i < batch_size; i++) {
    max_kv_len = std::max(max_kv_len, h_kv_lens[i]);
  }

  const bool use_partition = (max_kv_len > (IdType)kv_chunk_threshold);

  auto fill_params = [&](TQ4Params& params) {
    params.q = q;
    params.paged_kv = paged_kv;
    params.o = o;
    params.lse = nullptr;
    params.num_qo_heads = num_qo_heads;
    params.q_stride_n = num_qo_heads * HEAD_DIM;
    params.q_stride_h = HEAD_DIM;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = sm_scale;
    params.block_valid_mask = nullptr;
  };

  if (!use_partition) {
    std::vector<IdType> identity(batch_size);
    for (uint32_t i = 0; i < batch_size; i++) identity[i] = i;

    IdType* d_req_idx = nullptr;
    IdType* d_tile_idx = nullptr;
    IdType* d_chunk_size = nullptr;
    cudaMallocAsync(&d_req_idx, batch_size * sizeof(IdType), stream);
    cudaMallocAsync(&d_tile_idx, batch_size * sizeof(IdType), stream);
    cudaMallocAsync(&d_chunk_size, sizeof(IdType), stream);
    cudaMemcpyAsync(d_req_idx, identity.data(), batch_size * sizeof(IdType),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tile_idx, identity.data(), batch_size * sizeof(IdType),
                    cudaMemcpyHostToDevice, stream);
    IdType zero = 0;
    cudaMemcpyAsync(d_chunk_size, &zero, sizeof(IdType), cudaMemcpyHostToDevice, stream);

    TQ4Params params;
    fill_params(params);
    params.padded_batch_size = batch_size;
    params.partition_kv = false;
    params.request_indices = d_req_idx;
    params.kv_tile_indices = d_tile_idx;
    params.kv_chunk_size_ptr = d_chunk_size;
    params.o_indptr = nullptr;

    auto result = BatchDecodeWithPagedKVCacheTQ4Dispatched<
        HEAD_DIM, AttentionVariant, TQ4Params>(params, nullptr, nullptr, stream);

    cudaFreeAsync(d_req_idx, stream);
    cudaFreeAsync(d_tile_idx, stream);
    cudaFreeAsync(d_chunk_size, stream);
    return result;
  }

  // Partition-kv path
  std::vector<IdType> h_num_chunks(batch_size);
  uint32_t total_tiles = 0;
  for (uint32_t i = 0; i < batch_size; i++) {
    h_num_chunks[i] = std::max((IdType)1,
        (h_kv_lens[i] + (IdType)kv_chunk_size - 1) / (IdType)kv_chunk_size);
    total_tiles += h_num_chunks[i];
  }

  std::vector<IdType> h_req_idx(total_tiles), h_tile_idx(total_tiles);
  std::vector<IdType> h_o_indptr(batch_size + 1);
  h_o_indptr[0] = 0;
  uint32_t off = 0;
  for (uint32_t i = 0; i < batch_size; i++) {
    for (IdType t = 0; t < h_num_chunks[i]; t++) {
      h_req_idx[off + t] = i;
      h_tile_idx[off + t] = t;
    }
    off += h_num_chunks[i];
    h_o_indptr[i + 1] = off;
  }

  IdType* d_req_idx = nullptr;
  IdType* d_tile_idx = nullptr;
  IdType* d_o_indptr = nullptr;
  IdType* d_chunk_size_ptr = nullptr;
  float* d_lse = nullptr;
  DTypeO* d_tmp_v = nullptr;

  cudaMallocAsync(&d_req_idx, total_tiles * sizeof(IdType), stream);
  cudaMallocAsync(&d_tile_idx, total_tiles * sizeof(IdType), stream);
  cudaMallocAsync(&d_o_indptr, (batch_size + 1) * sizeof(IdType), stream);
  cudaMallocAsync(&d_chunk_size_ptr, sizeof(IdType), stream);
  cudaMallocAsync(&d_tmp_v, total_tiles * num_qo_heads * HEAD_DIM * sizeof(DTypeO), stream);
  cudaMallocAsync(&d_lse, total_tiles * num_qo_heads * sizeof(float), stream);

  cudaMemcpyAsync(d_req_idx, h_req_idx.data(), total_tiles * sizeof(IdType),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_tile_idx, h_tile_idx.data(), total_tiles * sizeof(IdType),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_o_indptr, h_o_indptr.data(), (batch_size + 1) * sizeof(IdType),
                  cudaMemcpyHostToDevice, stream);
  IdType h_cs = kv_chunk_size;
  cudaMemcpyAsync(d_chunk_size_ptr, &h_cs, sizeof(IdType), cudaMemcpyHostToDevice, stream);

  TQ4Params params;
  fill_params(params);
  params.lse = d_lse;
  params.padded_batch_size = total_tiles;
  params.partition_kv = true;
  params.request_indices = d_req_idx;
  params.kv_tile_indices = d_tile_idx;
  params.o_indptr = d_o_indptr;
  params.kv_chunk_size_ptr = d_chunk_size_ptr;

  auto result = BatchDecodeWithPagedKVCacheTQ4Dispatched<
      HEAD_DIM, AttentionVariant, TQ4Params>(params, d_tmp_v, d_lse, stream);

  cudaFreeAsync(d_req_idx, stream);
  cudaFreeAsync(d_tile_idx, stream);
  cudaFreeAsync(d_o_indptr, stream);
  cudaFreeAsync(d_chunk_size_ptr, stream);
  cudaFreeAsync(d_tmp_v, stream);
  cudaFreeAsync(d_lse, stream);
  return result;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_BATCH_DECODE_TQ4_RUN_CUH_
