/*
 * Batch decode with automatic partition-kv planning.
 *
 * Template function that handles:
 * 1. Host-side partition planning (chunk/tile allocation)
 * 2. GPU dispatch via BatchDecodeWithPagedKVCacheDispatched
 * 3. Workspace allocation (cudaMallocAsync)
 *
 * This is the "batteries included" API — caller only provides paged_kv + q.
 * For pre-allocated workspace, use the lower-level Dispatched API directly.
 */

#ifndef FLASHINFER_ATTENTION_BATCH_DECODE_RUN_CUH_
#define FLASHINFER_ATTENTION_BATCH_DECODE_RUN_CUH_

#include <algorithm>
#include <vector>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/page.cuh>

namespace flashinfer {

/*!
 * \brief Batch decode with paged KV cache, automatic partition-kv.
 * \tparam HEAD_DIM Head dimension (compile-time constant for kernel dispatch)
 * \tparam DTypeQ Query/output dtype (e.g. nv_bfloat16)
 * \tparam DTypeKV KV cache dtype
 * \tparam DTypeO Output dtype
 * \tparam IdType Index type (e.g. int32_t)
 * \tparam AttentionVariant Attention variant (e.g. DefaultAttention<...>)
 * \param q [batch_size, num_qo_heads * HEAD_DIM] query tensor
 * \param paged_kv Paged KV cache descriptor
 * \param o [batch_size, num_qo_heads * HEAD_DIM] output tensor
 * \param h_kv_lens Host array [batch_size] of KV lengths per sequence
 * \param batch_size Number of sequences
 * \param num_qo_heads Number of query/output heads
 * \param sm_scale Softmax scaling factor (1/sqrt(head_dim))
 * \param stream CUDA stream
 * \param kv_chunk_threshold Context length threshold for partition-kv (default 256)
 * \param kv_chunk_size Chunk size for partitioning (default 256)
 */
template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename IdType, typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheRun(
    DTypeQ* q,
    paged_kv_t<DTypeKV, IdType> paged_kv,
    DTypeO* o,
    const IdType* h_kv_lens,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    float sm_scale,
    cudaStream_t stream,
    uint32_t kv_chunk_threshold = 256,
    uint32_t kv_chunk_size = 256) {

  using BatchP = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;

  // Decide: partition-kv or non-partition based on max KV length
  IdType max_kv_len = 0;
  for (uint32_t i = 0; i < batch_size; i++) {
    max_kv_len = std::max(max_kv_len, h_kv_lens[i]);
  }

  const bool use_partition = (max_kv_len > (IdType)kv_chunk_threshold);

  if (!use_partition) {
    // ─── Non-partition path (short context) ───
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

    BatchP params(q, nullptr, paged_kv, o, nullptr, nullptr,
                  num_qo_heads, num_qo_heads * HEAD_DIM, HEAD_DIM,
                  -1, 0.0f, sm_scale, 1.0f, 1.0f);
    params.padded_batch_size = batch_size;
    params.partition_kv = false;
    params.request_indices = d_req_idx;
    params.kv_tile_indices = d_tile_idx;
    params.block_valid_mask = nullptr;
    params.o_indptr = nullptr;
    params.kv_chunk_size_ptr = d_chunk_size;

    auto result = BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, PosEncodingMode::kNone, AttentionVariant, BatchP>(
        params, nullptr, nullptr, false, stream);

    cudaFreeAsync(d_req_idx, stream);
    cudaFreeAsync(d_tile_idx, stream);
    cudaFreeAsync(d_chunk_size, stream);
    return result;
  }

  // ─── Partition-kv path (long context) ───
  std::vector<IdType> h_num_chunks(batch_size);
  uint32_t total_tiles = 0;
  for (uint32_t i = 0; i < batch_size; i++) {
    h_num_chunks[i] = (h_kv_lens[i] + kv_chunk_size - 1) / kv_chunk_size;
    if (h_num_chunks[i] == 0) h_num_chunks[i] = 1;
    total_tiles += h_num_chunks[i];
  }

  std::vector<IdType> h_req_idx(total_tiles);
  std::vector<IdType> h_tile_idx(total_tiles);
  std::vector<IdType> h_o_indptr(batch_size + 1);
  h_o_indptr[0] = 0;
  uint32_t offset = 0;
  for (uint32_t i = 0; i < batch_size; i++) {
    for (IdType t = 0; t < h_num_chunks[i]; t++) {
      h_req_idx[offset + t] = i;
      h_tile_idx[offset + t] = t;
    }
    offset += h_num_chunks[i];
    h_o_indptr[i + 1] = offset;
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
  IdType h_chunk = kv_chunk_size;
  cudaMemcpyAsync(d_chunk_size_ptr, &h_chunk, sizeof(IdType), cudaMemcpyHostToDevice, stream);

  BatchP params(q, nullptr, paged_kv, o, nullptr, nullptr,
                num_qo_heads, num_qo_heads * HEAD_DIM, HEAD_DIM,
                -1, 0.0f, sm_scale, 1.0f, 1.0f);
  params.padded_batch_size = total_tiles;
  params.partition_kv = true;
  params.request_indices = d_req_idx;
  params.kv_tile_indices = d_tile_idx;
  params.o_indptr = d_o_indptr;
  params.kv_chunk_size_ptr = d_chunk_size_ptr;
  params.block_valid_mask = nullptr;
  params.lse = d_lse;

  auto result = BatchDecodeWithPagedKVCacheDispatched<
      HEAD_DIM, PosEncodingMode::kNone, AttentionVariant, BatchP>(
      params, d_tmp_v, d_lse, false, stream);

  cudaFreeAsync(d_req_idx, stream);
  cudaFreeAsync(d_tile_idx, stream);
  cudaFreeAsync(d_o_indptr, stream);
  cudaFreeAsync(d_chunk_size_ptr, stream);
  cudaFreeAsync(d_tmp_v, stream);
  cudaFreeAsync(d_lse, stream);
  return result;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_BATCH_DECODE_RUN_CUH_
