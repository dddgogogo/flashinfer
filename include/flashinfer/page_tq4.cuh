/*
 * TurboQuant 4-bit (TQ4) paged KV cache support for FlashInfer.
 *
 * TQ4 encodes each KV head vector as:
 *   [d/2 packed 4-bit centroid indices] [4-byte F32 norm]
 * Total per row: d/2 + 4 bytes (68 bytes for d=128).
 *
 * The attention kernel operates in the rotated domain:
 *   - Q is pre-rotated by Pi^T before attention
 *   - K/V are stored as rotated+quantized centroids * norm
 *   - O is post-rotated by Pi after attention
 * This avoids per-token [d,d] matmul — only 2 small matmuls total.
 */
#ifndef FLASHINFER_PAGE_TQ4_CUH_
#define FLASHINFER_PAGE_TQ4_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utils.cuh"

namespace flashinfer {

/*!
 * \brief Paged KV cache with TQ4 (TurboQuant 4-bit) encoding.
 *
 * Unlike paged_kv_t<DType> which uses element-based strides, this struct
 * uses byte-based addressing since TQ4 rows have variable-width packing.
 *
 * Layout (HND): [max_pages, num_heads, page_size, row_bytes]
 * where row_bytes = head_dim/2 + 4
 *
 * Each row stores:
 *   [0 .. head_dim/2-1]: packed 4-bit centroid indices (2 per byte, LSB first)
 *   [head_dim/2 .. head_dim/2+3]: F32 norm of the original vector
 */
template <typename IdType>
struct paged_kv_tq4_t {
  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t row_bytes;     // head_dim/2 + 4
  size_t stride_page;     // num_heads * page_size * row_bytes (bytes)
  size_t stride_h;        // page_size * row_bytes (bytes, HND)
  size_t stride_n;        // row_bytes (bytes, within a head)

  uint8_t* k_data;        // byte-addressed page pool for K
  uint8_t* v_data;        // byte-addressed page pool for V
  const float* centroids; // [16] Lloyd-Max centroid values

  IdType* indices;        // [nnz_pages] physical page indices
  IdType* indptr;         // [batch_size + 1] CSR-style page pointers
  IdType* last_page_len;  // [batch_size] tokens in last page
  IdType* rope_pos_offset;// [batch_size] RoPE position offsets (optional)

  __host__ __device__ __forceinline__ paged_kv_tq4_t()
      : num_heads(0),
        page_size(),
        head_dim(0),
        batch_size(0),
        row_bytes(0),
        stride_page(0),
        stride_h(0),
        stride_n(0),
        k_data(nullptr),
        v_data(nullptr),
        centroids(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  __host__ __forceinline__ paged_kv_tq4_t(uint32_t num_heads, uint32_t page_size,
                                           uint32_t head_dim, uint32_t batch_size,
                                           uint8_t* k_data, uint8_t* v_data,
                                           const float* centroids, IdType* indices,
                                           IdType* indptr, IdType* last_page_len,
                                           IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        row_bytes(head_dim / 2 + 4),
        k_data(k_data),
        v_data(v_data),
        centroids(centroids),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    // HND layout: [page, head, token, packed_data]
    stride_n = (size_t)row_bytes;
    stride_h = (size_t)page_size * row_bytes;
    stride_page = (size_t)num_heads * page_size * row_bytes;
  }

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  /*!
   * \brief Get byte offset to the start of a TQ4 row in the page pool.
   * \param page_idx Physical page index
   * \param head_idx KV head index
   * \param entry_idx Token position within the page
   * \return Byte offset into k_data or v_data
   */
  __host__ __device__ __forceinline__ size_t get_row_offset(size_t page_idx, size_t head_idx,
                                                             size_t entry_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n;
  }

  /*!
   * \brief Get byte offset with bounds protection (returns 0 for out-of-bounds).
   */
  __device__ __forceinline__ size_t protective_get_row_offset(IdType page_iter, uint32_t head_idx,
                                                               uint32_t entry_idx,
                                                               IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_row_offset(__ldg(indices + page_iter), head_idx, entry_idx);
    } else {
      return 0;
    }
  }

  /*!
   * \brief Get pointer to packed indices of a K row.
   */
  __device__ __forceinline__ const uint8_t* get_k_packed_ptr(IdType page_iter, uint32_t head_idx,
                                                              uint32_t entry_idx) const {
    return k_data + get_row_offset(__ldg(indices + page_iter), head_idx, entry_idx);
  }

  /*!
   * \brief Get the F32 norm stored at the end of a K row.
   */
  __device__ __forceinline__ float get_k_norm(IdType page_iter, uint32_t head_idx,
                                               uint32_t entry_idx) const {
    const uint8_t* row = get_k_packed_ptr(page_iter, head_idx, entry_idx);
    return *reinterpret_cast<const float*>(row + head_dim / 2);
  }

  /*!
   * \brief Get pointer to packed indices of a V row.
   */
  __device__ __forceinline__ const uint8_t* get_v_packed_ptr(IdType page_iter, uint32_t head_idx,
                                                              uint32_t entry_idx) const {
    return v_data + get_row_offset(__ldg(indices + page_iter), head_idx, entry_idx);
  }

  /*!
   * \brief Get the F32 norm stored at the end of a V row.
   */
  __device__ __forceinline__ float get_v_norm(IdType page_iter, uint32_t head_idx,
                                               uint32_t entry_idx) const {
    const uint8_t* row = get_v_packed_ptr(page_iter, head_idx, entry_idx);
    return *reinterpret_cast<const float*>(row + head_dim / 2);
  }
};

// ─── Append Kernel: Quantize BF16 → TQ4 and write to paged KV ───

/*!
 * \brief Append one token per sequence to TQ4 paged KV cache.
 *
 * Input key/value are pre-processed: normalized to unit vectors and rotated by Pi^T.
 * This kernel quantizes each element to the nearest centroid, packs to 4-bit,
 * and stores packed data + norm to the page.
 *
 * Grid: (batch_size, num_heads, 1)
 * Block: (head_dim/2, 1, 1)
 *   - threadIdx.x handles 2 elements (1 packed byte)
 *   - blockIdx.y selects the head
 */
template <uint32_t head_dim, typename IdType>
__global__ void AppendPagedKVCacheTQ4DecodeKernel(
    paged_kv_tq4_t<IdType> paged_kv,
    const float* __restrict__ key_rotated,     // [batch, num_heads, head_dim] F32 unit vectors
    const float* __restrict__ value_rotated,    // [batch, num_heads, head_dim] F32 unit vectors
    const float* __restrict__ k_norms,          // [batch, num_heads] F32
    const float* __restrict__ v_norms) {        // [batch, num_heads] F32

  const uint32_t tx = threadIdx.x;  // 0 .. head_dim/2-1
  const uint32_t head_idx = blockIdx.y;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t num_heads = paged_kv.num_heads;

  // Find page and entry for the append position
  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_len[batch_idx];
  uint32_t page_iter = paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size;
  uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;
  IdType phys_page = __ldg(paged_kv.indices + page_iter);

  // Load centroids to shared memory
  __shared__ float s_centroids[16];
  if (tx < 16) {
    s_centroids[tx] = paged_kv.centroids[tx];
  }
  __syncthreads();

  // --- Encode K ---
  {
    size_t row_offset = paged_kv.get_row_offset(phys_page, head_idx, entry_idx);
    uint8_t* dst = paged_kv.k_data + row_offset;

    // Load 2 rotated elements
    uint32_t elem_base = tx * 2;
    const float* k_src = key_rotated + (batch_idx * num_heads + head_idx) * head_dim;
    float v0 = k_src[elem_base];
    float v1 = k_src[elem_base + 1];

    // Find nearest centroid for each element
    uint8_t best0 = 0, best1 = 0;
    float best_dist0 = fabsf(v0 - s_centroids[0]);
    float best_dist1 = fabsf(v1 - s_centroids[0]);
    #pragma unroll
    for (uint8_t c = 1; c < 16; ++c) {
      float d0 = fabsf(v0 - s_centroids[c]);
      float d1 = fabsf(v1 - s_centroids[c]);
      if (d0 < best_dist0) { best_dist0 = d0; best0 = c; }
      if (d1 < best_dist1) { best_dist1 = d1; best1 = c; }
    }

    // Pack 2 indices into 1 byte (LSB first)
    dst[tx] = (best1 << 4) | best0;

    // Thread 0 writes the norm
    if (tx == 0) {
      float norm = k_norms[batch_idx * num_heads + head_idx];
      *reinterpret_cast<float*>(dst + head_dim / 2) = norm;
    }
  }

  // --- Encode V ---
  {
    size_t row_offset = paged_kv.get_row_offset(phys_page, head_idx, entry_idx);
    uint8_t* dst = paged_kv.v_data + row_offset;

    uint32_t elem_base = tx * 2;
    const float* v_src = value_rotated + (batch_idx * num_heads + head_idx) * head_dim;
    float v0 = v_src[elem_base];
    float v1 = v_src[elem_base + 1];

    uint8_t best0 = 0, best1 = 0;
    float best_dist0 = fabsf(v0 - s_centroids[0]);
    float best_dist1 = fabsf(v1 - s_centroids[0]);
    #pragma unroll
    for (uint8_t c = 1; c < 16; ++c) {
      float d0 = fabsf(v0 - s_centroids[c]);
      float d1 = fabsf(v1 - s_centroids[c]);
      if (d0 < best_dist0) { best_dist0 = d0; best0 = c; }
      if (d1 < best_dist1) { best_dist1 = d1; best1 = c; }
    }

    dst[tx] = (best1 << 4) | best0;

    if (tx == 0) {
      float norm = v_norms[batch_idx * num_heads + head_idx];
      *reinterpret_cast<float*>(dst + head_dim / 2) = norm;
    }
  }
}

/*!
 * \brief Append a contiguous token range for a single sequence into TQ4 paged KV cache.
 *
 * Input key/value are rotated by Pi^T but not yet normalized.
 * This kernel computes per-row L2 norms, normalizes, quantizes each element to the
 * nearest centroid and writes packed data + norm directly to the paged cache,
 * eliminating intermediate normalize and packed staging buffers.
 *
 * Grid: (seq_len, num_heads, 1)
 * Block: (head_dim / 2, 1, 1)
 */
template <uint32_t head_dim, typename IdType>
__global__ void AppendPagedKVCacheTQ4ContiguousKernel(
    uint8_t* __restrict__ k_data,
    uint8_t* __restrict__ v_data,
    const IdType* __restrict__ page_indices,    // [num_pages]
    const float* __restrict__ centroids,        // [16]
    const float* __restrict__ key_rotated,      // [seq_len, num_heads, head_dim] F32 rotated vectors
    const float* __restrict__ value_rotated,    // [seq_len, num_heads, head_dim] F32 rotated vectors
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t page_size,
    uint32_t row_bytes,
    size_t stride_page,
    size_t stride_h,
    uint32_t token_offset) {
  const uint32_t token_idx = blockIdx.x;
  const uint32_t head_idx = blockIdx.y;
  const uint32_t tx = threadIdx.x;

  if (token_idx >= seq_len || head_idx >= num_heads) {
    return;
  }

  __shared__ float s_centroids[16];
  if (tx < 16) {
    s_centroids[tx] = centroids[tx];
  }
  __syncthreads();

  const uint32_t row_idx = token_idx * num_heads + head_idx;
  const uint32_t abs_token = token_offset + token_idx;
  const uint32_t page_iter = abs_token / page_size;
  const uint32_t entry_idx = abs_token % page_size;
  const IdType phys_page = __ldg(page_indices + page_iter);

  const size_t row_offset = (size_t)phys_page * stride_page + (size_t)head_idx * stride_h +
                            (size_t)entry_idx * row_bytes;
  const uint32_t elem_base = tx * 2;
  const uint32_t lane = tx & 31;
  const uint32_t warp_id = tx >> 5;

  __shared__ float s_k_inv_norm;
  __shared__ float s_v_inv_norm;
  __shared__ float s_k_norm;
  __shared__ float s_v_norm;
  __shared__ float s_k_partial[head_dim / 64];
  __shared__ float s_v_partial[head_dim / 64];

  const float* k_src = key_rotated + row_idx * head_dim;
  const float* v_src = value_rotated + row_idx * head_dim;
  const float k0_raw = k_src[elem_base];
  const float k1_raw = k_src[elem_base + 1];
  const float v0_raw = v_src[elem_base];
  const float v1_raw = v_src[elem_base + 1];

  float k_sq = k0_raw * k0_raw + k1_raw * k1_raw;
  float v_sq = v0_raw * v0_raw + v1_raw * v1_raw;
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
    v_sq += __shfl_down_sync(0xffffffff, v_sq, offset);
  }
  if (lane == 0) {
    s_k_partial[warp_id] = k_sq;
    s_v_partial[warp_id] = v_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float k_block_sq = lane < (head_dim / 64) ? s_k_partial[lane] : 0.f;
    float v_block_sq = lane < (head_dim / 64) ? s_v_partial[lane] : 0.f;
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      k_block_sq += __shfl_down_sync(0xffffffff, k_block_sq, offset);
      v_block_sq += __shfl_down_sync(0xffffffff, v_block_sq, offset);
    }
    if (lane == 0) {
      s_k_norm = sqrtf(k_block_sq);
      s_v_norm = sqrtf(v_block_sq);
      s_k_inv_norm = s_k_norm > 0.f ? __fdividef(1.f, s_k_norm) : 0.f;
      s_v_inv_norm = s_v_norm > 0.f ? __fdividef(1.f, s_v_norm) : 0.f;
    }
  }
  __syncthreads();

  // --- Encode K ---
  {
    uint8_t* dst = k_data + row_offset;
    float v0 = k0_raw * s_k_inv_norm;
    float v1 = k1_raw * s_k_inv_norm;

    uint8_t best0 = 0, best1 = 0;
    float best_dist0 = fabsf(v0 - s_centroids[0]);
    float best_dist1 = fabsf(v1 - s_centroids[0]);
#pragma unroll
    for (uint8_t c = 1; c < 16; ++c) {
      float d0 = fabsf(v0 - s_centroids[c]);
      float d1 = fabsf(v1 - s_centroids[c]);
      if (d0 < best_dist0) {
        best_dist0 = d0;
        best0 = c;
      }
      if (d1 < best_dist1) {
        best_dist1 = d1;
        best1 = c;
      }
    }

    dst[tx] = (best1 << 4) | best0;
    if (tx == 0) {
      *reinterpret_cast<float*>(dst + head_dim / 2) = s_k_norm;
    }
  }

  // --- Encode V ---
  {
    uint8_t* dst = v_data + row_offset;
    float v0 = v0_raw * s_v_inv_norm;
    float v1 = v1_raw * s_v_inv_norm;

    uint8_t best0 = 0, best1 = 0;
    float best_dist0 = fabsf(v0 - s_centroids[0]);
    float best_dist1 = fabsf(v1 - s_centroids[0]);
#pragma unroll
    for (uint8_t c = 1; c < 16; ++c) {
      float d0 = fabsf(v0 - s_centroids[c]);
      float d1 = fabsf(v1 - s_centroids[c]);
      if (d0 < best_dist0) {
        best_dist0 = d0;
        best0 = c;
      }
      if (d1 < best_dist1) {
        best_dist1 = d1;
        best1 = c;
      }
    }

    dst[tx] = (best1 << 4) | best0;
    if (tx == 0) {
      *reinterpret_cast<float*>(dst + head_dim / 2) = s_v_norm;
    }
  }
}

/*!
 * \brief Host launcher for TQ4 paged KV append (decode phase).
 */
template <typename IdType>
cudaError_t AppendPagedKVCacheTQ4Decode(paged_kv_tq4_t<IdType> paged_kv,
                                         const float* key_rotated,
                                         const float* value_rotated,
                                         const float* k_norms,
                                         const float* v_norms,
                                         cudaStream_t stream) {
  constexpr uint32_t HEAD_DIM = 256;  // Must match model head_dim (Qwen3.5 = 256)
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;

  dim3 nblks(batch_size, num_heads);    // One block per (batch, head)
  dim3 nthrs(HEAD_DIM / 2);            // head_dim/2 threads per block

  auto kernel = AppendPagedKVCacheTQ4DecodeKernel<HEAD_DIM, IdType>;
  kernel<<<nblks, nthrs, 0, stream>>>(paged_kv, key_rotated, value_rotated, k_norms, v_norms);
  return cudaGetLastError();
}

/*!
 * \brief Append a contiguous token range for one sequence into TQ4 paged KV cache.
 */
template <typename IdType>
cudaError_t AppendPagedKVCacheTQ4Contiguous(
    uint8_t* k_data, uint8_t* v_data, const IdType* page_indices, const float* centroids,
    const float* key_rotated, const float* value_rotated, uint32_t seq_len, uint32_t num_heads,
    uint32_t page_size, uint32_t head_dim, uint32_t token_offset, cudaStream_t stream = nullptr) {
  constexpr uint32_t HEAD_DIM = 256;
  if (head_dim != HEAD_DIM) {
    return cudaErrorInvalidValue;
  }

  dim3 nblks(seq_len, num_heads);
  dim3 nthrs(HEAD_DIM / 2);
  uint32_t row_bytes = head_dim / 2 + 4;
  size_t stride_h = (size_t)page_size * row_bytes;
  size_t stride_page = (size_t)num_heads * page_size * row_bytes;

  auto kernel = AppendPagedKVCacheTQ4ContiguousKernel<HEAD_DIM, IdType>;
  kernel<<<nblks, nthrs, 0, stream>>>(
      k_data, v_data, page_indices, centroids, key_rotated, value_rotated, seq_len, num_heads,
      page_size, row_bytes, stride_page, stride_h, token_offset);
  return cudaGetLastError();
}

}  // namespace flashinfer

#endif  // FLASHINFER_PAGE_TQ4_CUH_
