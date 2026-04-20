/*
 * TQ4 (TurboQuant 4-bit) batch decode attention kernel for FlashInfer.
 *
 * This kernel computes attention directly on TQ4-encoded KV cache:
 *   - Q is pre-rotated by Pi^T before kernel launch (done via cuBLAS)
 *   - K/V are stored as packed 4-bit centroid indices + F32 norm per head per token
 *   - O is post-rotated by Pi after kernel completes (done via cuBLAS)
 *
 * The kernel replaces the cp_async KV loading path from decode.cuh with:
 *   load packed bytes → centroid lookup → norm scale → BF16 shared memory
 * Then reuses the standard compute_qk() and update_local_state() functions.
 *
 * Copyright (c) 2026 Ling-RL. Licensed under Apache 2.0.
 */
#ifndef FLASHINFER_DECODE_TQ4_CUH_
#define FLASHINFER_DECODE_TQ4_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../math.cuh"
#include "../page_tq4.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "decode.cuh"  // for compute_qk, update_local_state, sync_state
#include "state.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

// ─── Params ──────────────────────────────────────────────────────────

template <typename DTypeQ_, typename DTypeO_, typename IdType_>
struct BatchDecodeParamsTQ4 {
  using DTypeQ = DTypeQ_;
  using DTypeKV = nv_bfloat16;  // shared memory type after dequant
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;                    // pre-rotated: q_rot = Pi^T * q
  paged_kv_tq4_t<IdType> paged_kv;
  DTypeO* o;
  float* lse;
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;

  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;

  __device__ __host__ BatchDecodeParamsTQ4()
      : q(nullptr), paged_kv(), o(nullptr), lse(nullptr), padded_batch_size(0),
        num_qo_heads(0), q_stride_n(0), q_stride_h(0), window_left(-1),
        logits_soft_cap(0.f), sm_scale(0.f), request_indices(nullptr),
        kv_tile_indices(nullptr), o_indptr(nullptr), kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr), partition_kv(false) {}

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }
  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

// ─── TQ4 Dequant Helper ─────────────────────────────────────────────

/*!
 * \brief Load one TQ4 row from global memory and write dequantized BF16 to shared memory.
 *
 * Each thread (tx) handles vec_size=8 elements:
 *   - Loads 4 packed bytes (8 × 4-bit indices) from the TQ4 row
 *   - Looks up centroids in shared memory
 *   - Multiplies by the row's F32 norm
 *   - Writes 8 BF16 values to the standard smem layout
 *
 * \param packed_row Pointer to start of TQ4 row (head_dim/2 bytes packed + 4 bytes norm)
 * \param centroids_smem Pointer to 16 F32 centroids in shared memory
 * \param dst BF16 shared memory destination for this row
 * \param head_dim The head dimension (128)
 * \param tx Thread index in x dimension (0..bdx-1, where bdx = head_dim / vec_size)
 * \param valid Whether this token is within bounds
 */
template <uint32_t vec_size>
__device__ __forceinline__ void tq4_dequant_row_to_smem(
    const uint8_t* packed_row, const float* centroids_smem,
    nv_bfloat16* dst, uint32_t head_dim, uint32_t tx, bool valid) {
  if (!valid) {
    // Fill with zeros for out-of-bounds tokens
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      dst[tx * vec_size + i] = __float2bfloat16(0.f);
    }
    return;
  }

  // Load 4 packed bytes (8 elements × 4 bits each)
  uint32_t packed = *reinterpret_cast<const uint32_t*>(packed_row + tx * 4);

  // Load norm from end of row
  float norm = *reinterpret_cast<const float*>(packed_row + head_dim / 2);

  // Unpack: 4 bytes → 8 centroid lookups → scale by norm → BF16
  #pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
    float c0 = centroids_smem[byte_val & 0x0F] * norm;
    float c1 = centroids_smem[(byte_val >> 4) & 0x0F] * norm;
    dst[tx * vec_size + i * 2] = __float2bfloat16(c0);
    dst[tx * vec_size + i * 2 + 1] = __float2bfloat16(c1);
  }
}

// ─── Batch Decode Device Function ────────────────────────────────────

/*!
 * \brief TQ4 batch decode attention with paged KV cache.
 *
 * Fork of BatchDecodeWithPagedKVCacheDevice from decode.cuh with TQ4 KV loading.
 * The compute path (compute_qk, update_local_state, sync_state) is reused as-is.
 *
 * Key difference: Instead of cp_async loading raw KV elements, we:
 *   1. Compute page offset for each token (divmod)
 *   2. Load 4 packed bytes + F32 norm per thread from global memory
 *   3. Centroid lookup + norm scale → write BF16 to standard k_smem/v_smem
 * Then compute_qk / update_local_state operate on standard BF16 shared memory.
 */
template <uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx, uint32_t bdy,
          uint32_t bdz, typename AttentionVariant, typename Params>
__device__ __inline__ void BatchDecodeWithPagedKVCacheTQ4Device(
    const Params& params, uint8_t smem[],
    const uint32_t bx = blockIdx.x, const uint32_t by = blockIdx.y,
    const uint32_t tx = threadIdx.x, const uint32_t ty = threadIdx.y,
    const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;  // nv_bfloat16
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto& paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;

  if (block_valid_mask && !block_valid_mask[bx]) return;

  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  // ─── Shared memory layout ───
  // [centroids: 16 F32 = 64B] [k_smem: tile * head_dim BF16] [v_smem: same] [smem_md: merge]
  float* centroids_smem = reinterpret_cast<float*>(smem);
  constexpr uint32_t centroids_bytes = 16 * sizeof(float);  // 64 bytes

  constexpr uint32_t tile_size = tile_size_per_bdx * bdy * bdz;
  DTypeKV* k_smem = reinterpret_cast<DTypeKV*>(smem + centroids_bytes);
  DTypeKV* v_smem = k_smem + tile_size * head_dim;
  float* smem_md = reinterpret_cast<float*>(v_smem + tile_size * head_dim);

  // Load centroids to shared memory (once)
  if (ty == 0 && tz == 0 && tx < 16) {
    centroids_smem[tx] = paged_kv.centroids[tx];
  }

  // Construct variant once (DefaultAttention doesn't use smem_ptr for our config)
  AttentionVariant variant(params, batch_idx, smem);
  block.sync();

  // Load Q to registers (no RoPE — already applied before pre-rotation)
  vec_t<float, vec_size> q_vec;
  q_vec.cast_load(q + batch_idx * params.q_stride_n + qo_head_idx * params.q_stride_h +
                  tx * vec_size);

  // ─── Main loop over KV tiles ───
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];
  const uint32_t page_base = (uint32_t)paged_kv.indptr[batch_idx] * (uint32_t)paged_kv.page_size;

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];
  vec_t<float, vec_size> dummy_freq;  // unused (no RoPE)

  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size); ++iter) {
    const uint32_t iter_base = iter * tile_size;

    // ─── Load and dequant K/V tiles to shared memory ───
    // TQ4 K and V share the same page/entry mapping. Load both with one row-offset
    // computation so the later compute_qk / update_local_state phases can reuse the
    // already-populated shared-memory tiles.
    #pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      const uint32_t tile_idx = (j * bdz + tz) * bdy + ty;
      const uint32_t abs_token = chunk_start + iter_base + tile_idx;
      const bool valid = (iter_base + tile_idx < chunk_size);

      if (valid) {
        uint32_t q_page, r_entry;
        paged_kv.page_size.divmod(page_base + abs_token, q_page, r_entry);
        size_t row_off = paged_kv.protective_get_row_offset(
            q_page, kv_head_idx, r_entry, last_indptr);
        tq4_dequant_row_to_smem<vec_size>(
            paged_kv.k_data + row_off, centroids_smem,
            k_smem + tile_idx * head_dim, head_dim, tx, true);
        tq4_dequant_row_to_smem<vec_size>(
            paged_kv.v_data + row_off, centroids_smem,
            v_smem + tile_idx * head_dim, head_dim, tx, true);
      } else {
        tq4_dequant_row_to_smem<vec_size>(
            nullptr, centroids_smem,
            k_smem + tile_idx * head_dim, head_dim, tx, false);
        tq4_dequant_row_to_smem<vec_size>(
            nullptr, centroids_smem,
            v_smem + tile_idx * head_dim, head_dim, tx, false);
      }
    }
    block.sync();

    // ─── Compute QK (reused from decode.cuh) ───
    compute_qk<PosEncodingMode::kNone, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + tz * bdy * tile_size_per_bdx * head_dim,
        q_vec, dummy_freq,
        chunk_start + iter_base,
        iter_base, chunk_size, qo_head_idx, kv_head_idx,
        s, st, tx, ty, tz);
    block.sync();

    // ─── Update state with V (reused from decode.cuh) ───
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + tz * bdy * tile_size_per_bdx * head_dim, s, 0, st, tx);
    block.sync();
  }

  // ─── Sync state across warps ───
  sync_state<vec_size, bdx, bdy, bdz>(
      variant, st,
      reinterpret_cast<float*>(smem + centroids_bytes), smem_md, tx, ty, tz);

  // ─── Normalize output (divide by softmax denominator d) ───
  #pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st.o[i] = variant.OutputTransform(params, st.o[i], bx, /*qo_idx=*/0, qo_head_idx,
                                      st.m, st.d, /*scale=*/1.0f);
  }

  // ─── Write output ───
  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
}

// ─── Kernel Wrapper ──────────────────────────────────────────────────

template <uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx, uint32_t bdy,
          uint32_t bdz, typename AttentionVariant, typename Params>
__global__ void BatchDecodeWithPagedKVCacheTQ4Kernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeWithPagedKVCacheTQ4Device<tile_size_per_bdx, vec_size, bdx, bdy, bdz,
                                        AttentionVariant, Params>(params, smem);
}

// ─── Dispatcher ──────────────────────────────────────────────────────

template <uint32_t HEAD_DIM, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheTQ4Dispatched(Params params,
                                                      typename Params::DTypeO* tmp_v,
                                                      float* tmp_s,
                                                      cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;  // nv_bfloat16
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  // For BF16 smem: vec_size=8, bdx=16 for head_dim=128
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);

  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? 4U : 1U;
    constexpr uint32_t tile_size = tile_size_per_bdx * bdy * bdz;

    // Shared memory: centroids + K tile + V tile + merge data
    const uint32_t smem_size =
        16 * sizeof(float) +                           // centroids (64 bytes)
        2 * tile_size * HEAD_DIM * sizeof(DTypeKV) +   // K + V tiles
        2 * bdy * bdz * sizeof(float);                 // merge data (smem_md)

    auto kernel = BatchDecodeWithPagedKVCacheTQ4Kernel<
        tile_size_per_bdx, vec_size, bdx, bdy, bdz, AttentionVariant, Params>;

    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nblks(padded_batch_size, num_kv_heads);
    dim3 nthrs(bdx, bdy, bdz);

    if (tmp_v == nullptr) {
      // Non-partition path
      params.partition_kv = false;
      void* args[] = {(void*)&params};
      FLASHINFER_CUDA_CALL(
          cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      // Partition-kv path
      params.partition_kv = true;
      auto o_orig = params.o;
      auto lse_orig = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      void* args[] = {(void*)&params};
      FLASHINFER_CUDA_CALL(
          cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
          tmp_v, tmp_s, params.o_indptr, o_orig, lse_orig,
          params.paged_kv.batch_size, nullptr,
          num_qo_heads, HEAD_DIM, false, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_TQ4_CUH_
