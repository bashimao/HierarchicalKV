/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <mutex>
#include <thread>
#include <vector>
#include "merlin/types.cuh"
#include "merlin/utils.cuh"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace nv {
namespace merlin {

// Vector Type of digests for memory access.
using VecD_Load = byte16;
// Vector Type of digests for computation.
using VecD_Comp = byte4;

// Select from double buffer.
// If i % 2 == 0, select buffer 0, else buffer 1.
__forceinline__ __device__ int same_buf(int i) { return (i & 0x01) ^ 0; }
// If i % 2 == 0, select buffer 1, else buffer 0.
__forceinline__ __device__ int diff_buf(int i) { return (i & 0x01) ^ 1; }

template <typename K>
__forceinline__ __device__ D empty_digest() {
  const K hashed_key = Murmur3HashDevice(static_cast<K>(EMPTY_KEY));
  return static_cast<D>(hashed_key >> 32);
}

template <typename K>
__forceinline__ __device__ D get_digest(const K& key) {
  const K hashed_key = Murmur3HashDevice(key);
  return static_cast<D>(hashed_key >> 32);
}

// Get vector of digests for computation.
template <typename K>
__forceinline__ __device__ VecD_Comp digests_from_hashed(const K& hashed_key) {
  D digest = static_cast<D>(hashed_key >> 32);
  // Set every byte in VecD_Comp to `digest`.
  return static_cast<VecD_Comp>(__byte_perm(digest, digest, 0x0000));
}

template <typename K>
__forceinline__ __device__ VecD_Comp empty_digests() {
  D digest = empty_digest<K>();
  // Set every byte in VecD_Comp to `digest`.
  return static_cast<VecD_Comp>(__byte_perm(digest, digest, 0x0000));
}

// Position alignment.
template <uint32_t ALIGN_SIZE>
__forceinline__ __device__ uint32_t align_to(uint32_t& pos) {
  constexpr uint32_t MASK = 0xffffffffU - (ALIGN_SIZE - 1);
  return pos & MASK;
}

template <typename ElementType>
__forceinline__ __device__ void LDGSTS(ElementType* dst,
                                       const ElementType* src);

template <>
__forceinline__ __device__ void LDGSTS<byte>(byte* dst, const byte* src) {
  byte element = *src;
  *dst = element;
}

template <>
__forceinline__ __device__ void LDGSTS<byte2>(byte2* dst, const byte2* src) {
  byte2 element = *src;
  *dst = element;
}

// Require compute ability >= 8.0
template <typename ElementType>
__forceinline__ __device__ void LDGSTS(ElementType* dst,
                                       const ElementType* src) {
  __pipeline_memcpy_async(dst, src, sizeof(ElementType));
}

template <typename S, typename K, int BUCKET_SIZE = 128>
struct CopyScoreEmpty {
  __forceinline__ __device__ static S* get_base_ptr(K** keys_ptr, int offset) {
    return nullptr;
  }
  __forceinline__ __device__ static void ldg_sts(S* dst, const S* src) {}
  __forceinline__ __device__ static S lgs(const S* src) { return 0; }
  __forceinline__ __device__ static void stg(S* dst, const S score_) {}
};

template <typename S, typename K, int BUCKET_SIZE = 128>
struct CopyScoreByPassCache {
  __forceinline__ __device__ static S* get_base_ptr(K** keys_ptr, int offset) {
    return reinterpret_cast<S*>(keys_ptr[offset] + BUCKET_SIZE);
  }

  __forceinline__ __device__ static void ldg_sts(S* dst, const S* src) {
    LDGSTS<S>(dst, src);
  }

  __forceinline__ __device__ static S lgs(const S* src) { return src[0]; }

  __forceinline__ __device__ static void stg(S* dst, const S score_) {
    __stcs(dst, score_);
  }
};

template <typename VecV = byte16, int GROUP_SIZE = 16>
struct CopyValueOneGroup {
  __forceinline__ __device__ static void ldg_sts(int rank, VecV* dst,
                                                 const VecV* src, int dim) {
    int offset = rank;
    if (offset < dim) LDGSTS<VecV>(dst + offset, src + offset);
  }

  __forceinline__ __device__ static void lds_stg(int rank, VecV* dst,
                                                 const VecV* src, int dim) {
    int offset = rank;
    if (offset < dim) {
      VecV vec_v = src[offset];
      __stcs(dst + offset, vec_v);
    }
  }
};

template <typename VecV = byte16, int GROUP_SIZE = 16>
struct CopyValueTwoGroup {
  __forceinline__ __device__ static void ldg_sts(int rank, VecV* dst,
                                                 const VecV* src,
                                                 const int dim) {
    int offset = rank;
    LDGSTS<VecV>(dst + offset, src + offset);
    offset += GROUP_SIZE;
    if (offset < dim) LDGSTS<VecV>(dst + offset, src + offset);
  }

  __forceinline__ __device__ static void lds_stg(int rank, VecV* dst,
                                                 const VecV* src,
                                                 const int dim) {
    int offset = rank;
    const VecV vec_v = src[offset];
    __stcs(dst + offset, vec_v);
    offset += GROUP_SIZE;
    if (offset < dim) {
      const VecV vec_v = src[offset];
      __stcs(dst + offset, vec_v);
    }
  }
};

template <typename VecV = byte16, int GROUP_SIZE = 16>
struct CopyValueMultipleGroup {
  __forceinline__ __device__ static void ldg_sts(int rank, VecV* dst,
                                                 const VecV* src,
                                                 const int dim) {
    for (int offset = rank; offset < dim; offset += GROUP_SIZE) {
      LDGSTS<VecV>(dst + offset, src + offset);
    }
  }

  __forceinline__ __device__ static void lds_stg(int rank, VecV* dst,
                                                 const VecV* src,
                                                 const int dim) {
    for (int offset = rank; offset < dim; offset += GROUP_SIZE) {
      VecV vec_v = src[offset];
      __stcs(dst + offset, vec_v);
    }
  }

  __forceinline__ __device__ static void ldg_stg(int rank, VecV* dst,
                                                 const VecV* src,
                                                 const int dim) {
    for (int offset = rank; offset < dim; offset += GROUP_SIZE) {
      VecV vec_v = __ldcs(src + offset);
      __stcs(dst + offset, vec_v);
    }
  }
};

template <typename K, typename S>
__forceinline__ __device__ void evict_key_score(K* evicted_keys,
                                                S* evicted_scores,
                                                const uint32_t evict_idx,
                                                const K& key, const S& score) {
  // Cache with evict_first strategy.
  __stcs(evicted_keys + evict_idx, key);
  if (evicted_scores != nullptr) {
    __stcs(evicted_scores + evict_idx, score);
  }
}

template <typename K, typename V, typename S, typename BUCKET = Bucket<K, V, S>>
__forceinline__ __device__ void update_score_digest(
    K* bucket_keys_ptr, const uint32_t bucket_capacity, const uint32_t key_pos,
    const K& key, const S& score) {
  S* dst_score_ptr = BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
  D* dst_digest_ptr =
      BUCKET::digests(bucket_keys_ptr, bucket_capacity, key_pos);
  // Cache in L2 cache, bypass L1 Cache.
  __stcg(dst_digest_ptr, get_digest<K>(key));
  __stcg(dst_score_ptr, score);
}

template <class K, class V, class S>
__forceinline__ __device__ void update_score(Bucket<K, V, S>* __restrict bucket,
                                             const int key_pos,
                                             const S* __restrict scores,
                                             const int key_idx) {
  if (scores == nullptr) {
    bucket->scores(key_pos)->store(device_nano<S>(),
                                   cuda::std::memory_order_relaxed);
  } else {
    bucket->scores(key_pos)->store(scores[key_idx],
                                   cuda::std::memory_order_relaxed);
  }
  return;
}

template <class V, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ void copy_vector(
    cg::thread_block_tile<TILE_SIZE> const& g, const V* src, V* dst,
    const size_t dim) {
  for (auto i = g.thread_rank(); i < dim; i += g.size()) {
    dst[i] = src[i];
  }

  //  cuda::barrier<cuda::thread_scope_device> bar;
  //  init(&bar, 1);
  //  cuda::memcpy_async(g, dst, src, dim * sizeof(V), bar);
  //
  //  bar.arrive_and_wait();
}

template <class K, class V, class S>
__forceinline__ __device__ Bucket<K, V, S>* get_key_position(
    Bucket<K, V, S>* __restrict buckets, const K key, size_t& bkt_idx,
    size_t& start_idx, const size_t buckets_num, const size_t bucket_max_size) {
  const K hashed_key = Murmur3HashDevice(key);
  const size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  bkt_idx = global_idx / bucket_max_size;
  start_idx = global_idx % bucket_max_size;
  return buckets + bkt_idx;
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ OccupyResult find_without_lock(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, S>* __restrict__ bucket,
    const K desired_key, const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;

  unsigned vote = 0;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    expected_key = current_key->load(cuda::std::memory_order_relaxed);
    vote = g.ballot(desired_key == expected_key);
    if (vote) {
      src_lane = __ffs(vote) - 1;
      key_pos = g.shfl(key_pos, src_lane);
      return OccupyResult::DUPLICATE;
    }
    vote = g.ballot(expected_key == static_cast<K>(EMPTY_KEY));
    if (vote) break;
  }
  return OccupyResult::CONTINUE;
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__device__ __inline__ OccupyResult find_and_lock_when_vacant(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, S>* __restrict__ bucket,
    const K desired_key, const S desired_score, K& evicted_key,
    const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;
  AtomicScore<S>* current_score;

  K local_min_score_key = static_cast<K>(EMPTY_KEY);

  S local_min_score_val = MAX_SCORE;
  S temp_min_score_val = MAX_SCORE;
  int local_min_score_pos = -1;

  unsigned vote = 0;
  bool result = false;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    // Step 1: try find and lock the desired_key.
    do {
      expected_key = desired_key;
      result = current_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      vote = g.ballot(result);
      if (vote) {
        src_lane = __ffs(vote) - 1;
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::DUPLICATE;
      }
      vote = g.ballot(expected_key == static_cast<K>(LOCKED_KEY));
      if (vote) continue;
      vote = g.ballot(expected_key == static_cast<K>(EMPTY_KEY));
      if (vote) break;
    } while (vote != 0);

    // Step 2: (TBD)try find empty location.
    while (vote) {
      src_lane = __ffs(vote) - 1;
      if (src_lane == g.thread_rank()) {
        expected_key = static_cast<K>(EMPTY_KEY);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      }
      result = g.shfl(result, src_lane);
      if (result) {
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::OCCUPIED_EMPTY;
      }
      result = g.shfl((expected_key == desired_key ||
                       expected_key == static_cast<K>(LOCKED_KEY)),
                      src_lane);
      if (result) {
        return OccupyResult::CONTINUE;
      }
      vote -= ((unsigned(0x1)) << src_lane);
    }
  }

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_score = bucket->scores(key_pos);

    // Step 4: record min score location.
    temp_min_score_val = current_score->load(cuda::std::memory_order_relaxed);
    if (temp_min_score_val < local_min_score_val) {
      expected_key =
          bucket->keys(key_pos)->load(cuda::std::memory_order_relaxed);
      if (expected_key != static_cast<K>(LOCKED_KEY) &&
          expected_key != static_cast<K>(EMPTY_KEY)) {
        local_min_score_key = expected_key;
        local_min_score_val = temp_min_score_val;
        local_min_score_pos = key_pos;
      }
    }
  }
  // Step 5: insert by evicting some one.
  const S global_min_score_val =
      cg::reduce(g, local_min_score_val, cg::less<S>());
  if (desired_score < global_min_score_val) {
    return OccupyResult::REFUSED;
  }
  vote = g.ballot(local_min_score_val <= global_min_score_val);
  if (vote) {
    src_lane = __ffs(vote) - 1;
    result = false;
    if (src_lane == g.thread_rank()) {
      // TBD: Here can be compare_exchange_weak. Do benchmark.
      current_key = bucket->keys(local_min_score_pos);
      current_score = bucket->scores(local_min_score_pos);
      evicted_key = local_min_score_key;
      result = current_key->compare_exchange_strong(
          local_min_score_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);

      // Need to recover when fail.
      if (result && (current_score->load(cuda::std::memory_order_relaxed) >
                     global_min_score_val)) {
        current_key->store(local_min_score_key,
                           cuda::std::memory_order_relaxed);
        result = false;
      }
    }
    result = g.shfl(result, src_lane);
    if (result) {
      // Not every `evicted_key` is correct expect the `src_lane` thread.
      key_pos = g.shfl(local_min_score_pos, src_lane);
      return (evicted_key == static_cast<K>(RECLAIM_KEY))
                 ? OccupyResult::OCCUPIED_RECLAIMED
                 : OccupyResult::EVICT;
    }
  }
  return OccupyResult::CONTINUE;
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ OccupyResult find_and_lock_when_full(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, S>* __restrict__ bucket,
    const K desired_key, const S desired_score, K& evicted_key,
    const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;
  AtomicScore<S>* current_score;

  K local_min_score_key = static_cast<K>(EMPTY_KEY);

  S local_min_score_val = MAX_SCORE;
  S temp_min_score_val = MAX_SCORE;
  int local_min_score_pos = -1;

  unsigned vote = 0;
  bool result = false;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    // Step 1: try find and lock the desired_key.
    do {
      expected_key = desired_key;
      result = current_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      vote = g.ballot(result);
      if (vote) {
        src_lane = __ffs(vote) - 1;
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::DUPLICATE;
      }
      vote = g.ballot(expected_key == static_cast<K>(LOCKED_KEY));
    } while (vote != 0);
  }

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    // Step 2: record min score location.
    temp_min_score_val =
        bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
    if (temp_min_score_val < local_min_score_val) {
      while ((expected_key = bucket->keys(key_pos)->load(
                  cuda::std::memory_order_relaxed)) ==
             static_cast<K>(LOCKED_KEY))
        ;
      local_min_score_key = expected_key;
      local_min_score_val = temp_min_score_val;
      local_min_score_pos = key_pos;
    }
  }

  // Step 3: insert by evicting some one.
  const S global_min_score_val =
      cg::reduce(g, local_min_score_val, cg::less<S>());
  if (desired_score < global_min_score_val) {
    return OccupyResult::REFUSED;
  }
  vote = g.ballot(local_min_score_val <= global_min_score_val);
  if (vote) {
    src_lane = __ffs(vote) - 1;
    result = false;
    if (src_lane == g.thread_rank()) {
      // TBD: Here can be compare_exchange_weak. Do benchmark.
      current_key = bucket->keys(local_min_score_pos);
      current_score = bucket->scores(local_min_score_pos);
      evicted_key = local_min_score_key;
      result = current_key->compare_exchange_strong(
          local_min_score_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);

      // Need to recover when fail.
      if (result && (current_score->load(cuda::std::memory_order_relaxed) >
                     global_min_score_val)) {
        current_key->store(local_min_score_key,
                           cuda::std::memory_order_relaxed);
        result = false;
      }
    }
    result = g.shfl(result, src_lane);
    if (result) {
      // Not every `evicted_key` is correct expect the `src_lane` thread.
      key_pos = g.shfl(local_min_score_pos, src_lane);
      return (evicted_key == static_cast<K>(RECLAIM_KEY))
                 ? OccupyResult::OCCUPIED_RECLAIMED
                 : OccupyResult::EVICT;
    }
  }
  return OccupyResult::CONTINUE;
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ OccupyResult find_and_lock_for_update(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, S>* __restrict__ bucket,
    const K desired_key, const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;

  unsigned vote = 0;
  bool result = false;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    // Step 1: try find and lock the desired_key.
    do {
      expected_key = desired_key;
      result = current_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      vote = g.ballot(result);
      if (vote) {
        src_lane = __ffs(vote) - 1;
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::DUPLICATE;
      }
      vote = g.ballot(expected_key == static_cast<K>(EMPTY_KEY));
      if (vote) return OccupyResult::REFUSED;
      vote = g.ballot(expected_key == static_cast<K>(LOCKED_KEY));
    } while (vote != 0);
  }
  return OccupyResult::REFUSED;
}

}  // namespace merlin
}  // namespace nv
