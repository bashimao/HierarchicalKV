/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/semaphore>
#include "merlin/types.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

template <class T>
using system_atomic = cuda::atomic<T, cuda::thread_scope_system>;
template <class T>
using device_atomic = cuda::atomic<T, cuda::thread_scope_device>;

using device_barrier = cuda::barrier<cuda::thread_scope_device>;

template <class K, cuda::thread_scope N_ERASURES_SCOPE>
inline __device__ void pref_inv_add(
    const K key, const uint32_t count, const uint32_t capacity,
    device_atomic<uint32_t>* const size,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    cuda::atomic<uint32_t, N_ERASURES_SCOPE>* const n_insertions,
    K* const inserted_keys) {
  assert(key != EMPTY_KEY && key != RECLAIM_KEY);

  using key_type = K;

  const uint32_t origin_slot_idx{
      static_cast<uint32_t>(Murmur3HashDevice(key) % capacity)};

  // Linearly probe to locate slot.
  key_type old_key;
  bool have_tombstone{};
  uint32_t slot_idx;
  for (slot_idx = origin_slot_idx;; slot_idx = (slot_idx + 1) % capacity) {
    if (have_tombstone) {
      // Scan...
      old_key = key_slots[slot_idx].load(cuda::memory_order_relaxed);
      if (old_key == key) {
        break;
      }

      // ... until we find the insertion point, ...
      if (old_key != EMPTY_KEY) {
        continue;
      }

      // ... and in that case we need to rescan.
      for (slot_idx = origin_slot_idx;; slot_idx = (slot_idx + 1) % capacity) {
        // Attempt to replace tombstone.
        old_key = RECLAIM_KEY;
        if (key_slots[slot_idx].compare_exchange_strong(
                old_key, key, cuda::memory_order_relaxed)) {
          num_tombstones->fetch_sub(1, cuda::memory_order_relaxed);
          break;
        }

        // Attempt to replace empty.
        assert(*size < capacity);
        old_key = EMPTY_KEY;
        if (key_slots[slot_idx].compare_exchange_strong(
                old_key, key, cuda::memory_order_relaxed)) {
          size->fetch_add(1, cuda::memory_order_relaxed);
          break;
        }
      }
      break;
    } else {
      // Scan and attempt to insert...
      assert(*size < capacity);
      old_key = EMPTY_KEY;
      if (key_slots[slot_idx].compare_exchange_strong(
              old_key, key, cuda::memory_order_relaxed)) {
        size->fetch_add(1, cuda::memory_order_relaxed);
        break;
      }
      if (old_key == key) {
        break;
      }

      // ... until we find a tombstone.
      have_tombstone = old_key == RECLAIM_KEY;
    }
  }

  // slot_idx now points to correct slot.
  assert(key_slots[slot_idx].load(cuda::memory_order_relaxed) == key);
  const uint32_t prev_refs{
      ref_slots[slot_idx].fetch_add(count, cuda::memory_order_relaxed)};
  if (prev_refs == 0 && n_insertions) {
    slot_idx = n_insertions->fetch_add(1, cuda::memory_order_relaxed);
    if (inserted_keys) {
      inserted_keys[slot_idx] = key;
    }
  }
}

template <class K, cuda::thread_scope N_ERASURES_SCOPE>
inline __device__ void pref_inv_drop(
    const K key, const uint32_t capacity,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    cuda::atomic<uint32_t, N_ERASURES_SCOPE>* const n_erasures,
    K* const erased_keys) {
  assert(key != EMPTY_KEY && key != RECLAIM_KEY);

  using key_type = K;

  const uint32_t origin_slot_idx{
      static_cast<uint32_t>(Murmur3HashDevice(key) % capacity)};

  // Linearly probe to locate slot.
  key_type old_key;
  uint32_t slot_idx;
  for (slot_idx = origin_slot_idx;; slot_idx = (slot_idx + 1) % capacity) {
    // Scan until we either find the key or an empty slot.
    old_key = key_slots[slot_idx].load(cuda::memory_order_relaxed);
    if (old_key == key) {
      break;
    }

    // This should not happen if the cache is sufficiently sized.
    assert(old_key != EMPTY_KEY);
  }

  const uint32_t count{
      ref_slots[slot_idx].fetch_sub(1, cuda::memory_order_relaxed)};
  assert(count >= 1);
  if (count == 1) {
    key_slots[slot_idx].store(RECLAIM_KEY, cuda::memory_order_relaxed);
    num_tombstones->fetch_add(1, cuda::memory_order_relaxed);
    if (n_erasures) {
      slot_idx = n_erasures->fetch_add(1, cuda::memory_order_relaxed);
      if (erased_keys) {
        erased_keys[slot_idx] = key;
      }
    }
  }
}

template <class K>
inline __device__ bool pref_inv_contains(
    const K key, const uint32_t capacity,
    const device_atomic<K>* const key_slots) {
  assert(key != EMPTY_KEY && key != RECLAIM_KEY);

  using key_type = K;

  // Linearly probe to locate slot.
  key_type old_key;
  uint32_t slot_idx{static_cast<uint32_t>(Murmur3HashDevice(key) % capacity)};
  for (;; slot_idx = (slot_idx + 1) % capacity) {
    // Scan until we find the key, or ...
    old_key = key_slots[slot_idx].load(cuda::memory_order_relaxed);
    if (old_key == key) {
      break;
    }

    // ... until the insertion point.
    if (old_key == EMPTY_KEY) {
      return false;
    }
  }

  return true;
}

template <class K>
inline __device__ uint32_t
pref_inv_get(const K key, const uint32_t capacity,
             const device_atomic<K>* const key_slots,
             const device_atomic<uint32_t>* const ref_slots) {
  assert(key != EMPTY_KEY && key != RECLAIM_KEY);

  using key_type = K;

  // Linearly probe to locate slot.
  key_type old_key;
  uint32_t slot_idx{static_cast<uint32_t>(Murmur3HashDevice(key) % capacity)};
  for (;; slot_idx = (slot_idx + 1) % capacity) {
    // Scan until we find the key, or ...
    old_key = key_slots[slot_idx].load(cuda::memory_order_relaxed);
    if (old_key == key) {
      break;
    }

    // ... until the insertion point.
    if (old_key == EMPTY_KEY) {
      return 0;
    }
  }

  return ref_slots[slot_idx];
}

template <class K>
__global__ void pref_inv_init_kernel(
    const uint32_t capacity, device_atomic<uint32_t>* const size,
    device_atomic<uint32_t>* const num_reclaimed,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    device_barrier* const cleanup_barrier) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};

  if (tid == 0) {
    size->store(0, cuda::memory_order_relaxed);
    num_reclaimed->store(0, cuda::memory_order_relaxed);
    init(cleanup_barrier, capacity + 1);
  }

  if (tid < capacity) {
    key_slots[tid].store(EMPTY_KEY, cuda::memory_order_relaxed);
    ref_slots[tid].store(0, cuda::memory_order_relaxed);
  }
}

template <class K>
__global__ void pref_inv_cleanup_kernel(
    const uint32_t capacity, device_atomic<uint32_t>* const size,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    device_barrier* const cleanup_sync) {
  using key_type = K;

  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < capacity) {
    // Fetch and reset slot data, and wait until all other threads are ready.
    const key_type key{
        key_slots[tid].exchange(EMPTY_KEY, cuda::memory_order_relaxed)};
    const uint32_t cnt{ref_slots[tid].exchange(0, cuda::memory_order_relaxed)};
    if (tid == 0) {
      size->store(0, cuda::memory_order_relaxed);
      num_tombstones->store(0, cuda::memory_order_relaxed);
    }
    cleanup_sync->arrive_and_wait();

    // Reinsert slot data.
    if (key != EMPTY_KEY && key != RECLAIM_KEY) {
      pref_inv_add(key, cnt, capacity, size, num_tombstones, key_slots,
                   ref_slots, reinterpret_cast<device_atomic<uint32_t>*>(0),
                   reinterpret_cast<key_type*>(0));
    }

    cleanup_sync->arrive();
  }
}

template <class K>
__global__ void pref_inv_invoke_cleanup_kernel(
    const uint32_t capacity, device_atomic<uint32_t>* const size,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    device_barrier* const cleanup_sync) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  assert(tid == 0);

  constexpr uint32_t block_size{512};
  const uint32_t grid_size{(capacity + block_size - 1) / block_size};
  pref_inv_cleanup_kernel<<<grid_size, block_size>>>(
      capacity, size, num_tombstones, key_slots, ref_slots, cleanup_sync);

  // Can do this more elegantly with `cudaStreamTailLaunch` with Hopper.
  cleanup_sync->arrive_and_wait();
  cleanup_sync->arrive_and_wait();
}

template <class K, cuda::thread_scope N_INSERTIONS_SCOPE>
__global__ void pref_inv_add_kernel(
    const K* const keys, const uint32_t num_keys, const uint32_t capacity,
    device_atomic<uint32_t>* const size,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    cuda::atomic<uint32_t, N_INSERTIONS_SCOPE>* const n_insertions,
    K* const inserted_keys) {
  assert(capacity - size->load(cuda::memory_order_relaxed) >= num_keys);

  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < num_keys) {
    pref_inv_add(keys[tid], 1, capacity, size, num_tombstones, key_slots,
                 ref_slots, n_insertions, inserted_keys);
  }
}

template <class K, cuda::thread_scope N_INSERTIONS_SCOPE>
__global__ void pref_inv_invoke_cleanup_add_kernel(
    const K* const keys, const uint32_t num_keys, const uint32_t capacity,
    device_atomic<uint32_t>* const size,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    device_barrier* const cleanup_sync,
    cuda::atomic<uint32_t, N_INSERTIONS_SCOPE>* const n_insertions,
    K* const inserted_keys, const uint32_t grid_size,
    const uint32_t block_size) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  assert(tid == 0);

  if (num_tombstones->load(cuda::memory_order_relaxed) > capacity / 4) {  // 25%
    const uint32_t grid_size{(capacity + block_size - 1) / block_size};
    pref_inv_cleanup_kernel<<<grid_size, block_size>>>(
        capacity, size, num_tombstones, key_slots, ref_slots, cleanup_sync);

    // Can do this more elegantly with `cudaStreamTailLaunch` with Hopper.
    cleanup_sync->arrive_and_wait();
    cleanup_sync->arrive_and_wait();
  }

  if (n_insertions) {
    n_insertions->store(0, cuda::memory_order_relaxed);
  }

  pref_inv_add_kernel<<<grid_size, block_size>>>(
      keys, num_keys, capacity, size, num_tombstones, key_slots, ref_slots,
      n_insertions, inserted_keys);
}

template <class K, cuda::thread_scope N_ERASURES_SCOPE>
__global__ void pref_inv_drop_kernel(
    const K* const keys, const uint32_t num_keys, const uint32_t capacity,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    cuda::atomic<uint32_t, N_ERASURES_SCOPE>* const n_erasures,
    K* const erased_keys) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < num_keys) {
    pref_inv_drop(keys[tid], capacity, num_tombstones, key_slots, ref_slots,
                  n_erasures, erased_keys);
  }
}

template <class K, cuda::thread_scope N_ERASURES_SCOPE>
__global__ void pref_inv_invoke_drop_kernel(
    const K* const keys, const uint32_t num_keys, const uint32_t capacity,
    device_atomic<uint32_t>* const num_tombstones,
    device_atomic<K>* const key_slots, device_atomic<uint32_t>* const ref_slots,
    cuda::atomic<uint32_t, N_ERASURES_SCOPE>* const n_erasures,
    K* const erased_keys, const uint32_t grid_size, const uint32_t block_size) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  assert(tid == 0);

  if (n_erasures) {
    n_erasures->store(0, cuda::memory_order_relaxed);
  }

  pref_inv_drop_kernel<<<grid_size, block_size>>>(
      keys, num_keys, capacity, num_tombstones, key_slots, ref_slots,
      n_erasures, erased_keys);
}

template <class K>
__global__ void pref_inv_keys_kernel(const uint32_t capacity,
                                     const device_atomic<K>* const key_slots,
                                     K* const keys,
                                     device_atomic<uint32_t>* const num_keys) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < capacity) {
    const auto key{key_slots[tid].load(cuda::memory_order_relaxed)};
    if (key != EMPTY_KEY && key != RECLAIM_KEY) {
      keys[num_keys->fetch_add(1, cuda::memory_order_relaxed)] = key;
    }
  }
}

template <class K>
__global__ void pref_inv_contains_kernel(
    const K* const keys, const uint32_t num_keys, const uint32_t capacity,
    const device_atomic<uint32_t>* const size,
    const device_atomic<K>* const key_slots, bool* const exists) {
  assert(size->load(cuda::memory_order_relaxed) < capacity);

  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < num_keys) {
    exists[tid] = pref_inv_contains(keys[tid], capacity, key_slots);
  }
}

template <class K>
__global__ void pref_inv_get_kernel(
    const K* const keys, const uint32_t num_keys, const uint32_t capacity,
    const device_atomic<uint32_t>* const size,
    const device_atomic<K>* const key_slots,
    const device_atomic<uint32_t>* const ref_slots, uint32_t* const num_refs) {
  assert(size->load(cuda::memory_order_relaxed) < capacity);

  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < num_keys) {
    num_refs[tid] = pref_inv_get(keys[tid], capacity, key_slots, ref_slots);
  }
}

/**
 * Forward declares required to make templated ostream overload work.
 */
template <class K>
class PrefetchInventory;

template <class K>
std::ostream& operator<<(std::ostream&, const PrefetchInventory<K>&);

/**
 * Fixed size GPU hashtable for reference counting.
 *
 * The inventory APIs are not thread-safe, if you do not use the same stream.
 */
template <class K>
class PrefetchInventory final {
 public:
  using key_type = K;
  using size_type = uint32_t;

  using atomic_key_type = device_atomic<key_type>;
  using atomic_size_type = device_atomic<size_type>;
  using barrier_type = device_barrier;

  PrefetchInventory() = delete;
  PrefetchInventory(const PrefetchInventory&) = delete;
  PrefetchInventory(PrefetchInventory&&) = delete;
  PrefetchInventory operator=(const PrefetchInventory&) = delete;
  PrefetchInventory operator=(PrefetchInventory&&) = delete;

  PrefetchInventory(size_type capacity, cudaStream_t stream);

  ~PrefetchInventory();

  inline void cleanup(cudaStream_t stream) {
    pref_inv_invoke_cleanup_kernel<<<1, 1, 0, stream>>>(
        capacity, d_size_, d_num_reclaimed_, d_key_slots_, d_ref_slots_,
        d_cleanup_sync_);
  }

  inline void add(const key_type* const dh_keys, const size_type n,
                  cudaStream_t stream) {
    add(dh_keys, n, nullptr, stream);
  }

  inline void add(const key_type* const dh_keys, const size_type n,
                  size_type* dh_n_insertions, cudaStream_t stream) {
    add(dh_keys, n, dh_n_insertions, nullptr, stream);
  }

  inline void add(const key_type* const dh_keys, size_type n,
                  size_type* const dh_n_insertions,
                  key_type* const dh_inserted_keys, cudaStream_t stream) {
    constexpr uint32_t block_size{512};
    const size_t grid_size{SAFE_GET_GRID_SIZE(n, block_size)};

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, dh_n_insertions));
    if (attr.type == cudaMemoryTypeDevice) {
      pref_inv_invoke_cleanup_add_kernel<<<1, 1, 0, stream>>>(
          dh_keys, n, capacity, d_size_, d_num_reclaimed_, d_key_slots_,
          d_ref_slots_, d_cleanup_sync_,
          reinterpret_cast<device_atomic<size_type>*>(dh_n_insertions),
          dh_inserted_keys, grid_size, block_size);

      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      pref_inv_invoke_cleanup_add_kernel<<<1, 1, 0, stream>>>(
          dh_keys, n, capacity, d_size_, d_num_reclaimed_, d_key_slots_,
          d_ref_slots_, d_cleanup_sync_,
          reinterpret_cast<system_atomic<size_type>*>(dh_n_insertions),
          dh_inserted_keys, grid_size, block_size);
    }
  }

  inline void drop(const key_type* const dh_keys, const size_type n,
                   cudaStream_t stream) {
    drop(dh_keys, n, nullptr, stream);
  }

  inline void drop(const key_type* const dh_keys, const size_type n,
                   size_type* const dh_n_erasures, cudaStream_t stream) {
    drop(dh_keys, n, dh_n_erasures, nullptr, stream);
  }

  inline void drop(const key_type* const dh_keys, const size_type n,
                   size_type* const dh_n_erasures, K* const dh_erased_keys,
                   cudaStream_t stream) {
    constexpr uint32_t block_size{512};
    const size_t grid_size{SAFE_GET_GRID_SIZE(n, block_size)};

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, dh_n_erasures));
    if (attr.type == cudaMemoryTypeDevice) {
      pref_inv_invoke_drop_kernel<<<1, 1, 0, stream>>>(
          dh_keys, n, capacity, d_num_reclaimed_, d_key_slots_, d_ref_slots_,
          reinterpret_cast<device_atomic<size_type>*>(dh_n_erasures),
          dh_erased_keys, grid_size, block_size);
    } else {
      pref_inv_invoke_drop_kernel<<<1, 1, 0, stream>>>(
          dh_keys, n, capacity, d_num_reclaimed_, d_key_slots_, d_ref_slots_,
          reinterpret_cast<system_atomic<size_type>*>(dh_n_erasures),
          dh_erased_keys, grid_size, block_size);
    }
  }

  size_type size(cudaStream_t stream) const;

  size_type keys(key_type* dh_keys, cudaStream_t stream) const;

  inline void contains(const key_type* const dh_keys, const size_type n,
                       bool* const dh_exists, cudaStream_t stream) const {
    constexpr uint32_t block_size{512};
    const size_t grid_size{SAFE_GET_GRID_SIZE(n, block_size)};
    pref_inv_contains_kernel<<<grid_size, block_size, 0, stream>>>(
        dh_keys, n, capacity, d_size_, d_key_slots_, dh_exists);
  }

  bool contains(key_type key, cudaStream_t stream) const;

  inline void get(const key_type* const dh_keys, const size_type n,
                  size_type* const dh_num_refs, cudaStream_t stream) const {
    constexpr uint32_t block_size{512};
    const size_t grid_size{SAFE_GET_GRID_SIZE(n, block_size)};
    pref_inv_get_kernel<<<grid_size, block_size, 0, stream>>>(
        dh_keys, n, capacity, d_size_, d_key_slots_, d_ref_slots_, dh_num_refs);
  }

  size_type get(key_type key, cudaStream_t stream) const;

  friend std::ostream& operator<<<K>(std::ostream&, const PrefetchInventory&);

 public:
  const size_type capacity;

 private:
  atomic_size_type* d_size_;
  atomic_size_type* d_num_reclaimed_;
  atomic_key_type* d_key_slots_;
  atomic_size_type* d_ref_slots_;
  barrier_type* d_cleanup_sync_;
};

template <class K>
PrefetchInventory<K>::PrefetchInventory(const size_type capacity,
                                        cudaStream_t stream)
    : capacity{capacity} {
  // Limit to max grid size (so we do not need an extra buffers for cleanup).
  assert(capacity < UINT32_C(1) << 31);

  CUDA_CHECK(cudaMallocAsync(&d_size_, sizeof(atomic_size_type), stream));
  CUDA_CHECK(
      cudaMallocAsync(&d_num_reclaimed_, sizeof(atomic_size_type), stream));
  CUDA_CHECK(cudaMallocAsync(&d_key_slots_, capacity * sizeof(atomic_key_type),
                             stream));
  CUDA_CHECK(cudaMallocAsync(&d_ref_slots_, capacity * sizeof(atomic_size_type),
                             stream));
  CUDA_CHECK(cudaMallocAsync(&d_cleanup_sync_, sizeof(barrier_type), stream));

  constexpr uint32_t block_size{512};
  const size_t grid_size{SAFE_GET_GRID_SIZE(capacity, block_size)};
  pref_inv_init_kernel<<<grid_size, block_size, 0, stream>>>(
      capacity, d_size_, d_num_reclaimed_, d_key_slots_, d_ref_slots_,
      d_cleanup_sync_);
}

template <class K>
PrefetchInventory<K>::~PrefetchInventory() {
  CUDA_CHECK(cudaFree(d_size_));
  d_size_ = nullptr;

  CUDA_CHECK(cudaFree(d_num_reclaimed_));
  d_num_reclaimed_ = nullptr;

  CUDA_CHECK(cudaFree(d_key_slots_));
  d_key_slots_ = nullptr;

  CUDA_CHECK(cudaFree(d_ref_slots_));
  d_ref_slots_ = nullptr;

  CUDA_CHECK(cudaFree(d_cleanup_sync_));
  d_cleanup_sync_ = nullptr;
}

template <class K>
PrefetchInventory<K>::size_type PrefetchInventory<K>::size(
    cudaStream_t stream) const {
  size_type size;
  CUDA_CHECK(cudaMemcpyAsync(&size, d_size_, sizeof(size_type),
                             cudaMemcpyDeviceToHost, stream));

  size_type num_reclaimed;
  CUDA_CHECK(cudaMemcpyAsync(&num_reclaimed, d_num_reclaimed_,
                             sizeof(size_type), cudaMemcpyDeviceToHost,
                             stream));

  return size - num_reclaimed;
}

template <class K>
PrefetchInventory<K>::size_type PrefetchInventory<K>::keys(
    key_type* const dh_keys, cudaStream_t stream) const {
  atomic_size_type* d_num_keys;
  CUDA_CHECK(cudaMallocAsync(&d_num_keys, sizeof(atomic_size_type), stream));
  CUDA_CHECK(cudaMemsetAsync(d_num_keys, 0, sizeof(atomic_size_type), stream));

  constexpr uint32_t block_size{512};
  const size_t grid_size{SAFE_GET_GRID_SIZE(capacity, block_size)};
  pref_inv_keys_kernel<<<grid_size, block_size, 0, stream>>>(
      capacity, d_key_slots_, dh_keys, d_num_keys);

  size_type num_keys;
  CUDA_CHECK(cudaMemcpyAsync(&num_keys, d_num_keys, sizeof(size_type),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_num_keys, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return num_keys;
}

template <class K>
bool PrefetchInventory<K>::contains(const key_type key,
                                    cudaStream_t stream) const {
  key_type* d_keys;
  CUDA_CHECK(cudaMallocAsync(&d_keys, sizeof(key_type), stream));
  CUDA_CHECK(cudaMemcpyAsync(d_keys, &key, sizeof(key_type),
                             cudaMemcpyHostToDevice, stream));

  bool* d_exists;
  CUDA_CHECK(cudaMallocAsync(&d_exists, sizeof(bool), stream));

  contains(d_keys, 1, d_exists, stream);
  CUDA_CHECK(cudaFreeAsync(d_keys, stream));

  bool exists;
  CUDA_CHECK(cudaMemcpyAsync(&exists, d_exists, sizeof(bool),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_exists, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return exists;
}

template <class K>
PrefetchInventory<K>::size_type PrefetchInventory<K>::get(
    const key_type key, cudaStream_t stream) const {
  key_type* d_keys;
  CUDA_CHECK(cudaMallocAsync(&d_keys, sizeof(key_type), stream));
  CUDA_CHECK(cudaMemcpyAsync(d_keys, &key, sizeof(key_type),
                             cudaMemcpyHostToDevice, stream));

  size_type* d_num_refs;
  CUDA_CHECK(cudaMallocAsync(&d_num_refs, sizeof(size_type), stream));

  get(d_keys, 1, d_num_refs, stream);
  CUDA_CHECK(cudaFreeAsync(d_keys, stream));

  size_type num_refs;
  CUDA_CHECK(cudaMemcpyAsync(&num_refs, d_num_refs, sizeof(size_type),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_num_refs, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return num_refs;
}

template <class K>
std::ostream& operator<<(std::ostream& os, const PrefetchInventory<K>& inv) {
  using key_type = typename PrefetchInventory<K>::key_type;
  using size_type = typename PrefetchInventory<K>::size_type;

  for (size_type i{}; i < 80; ++i) {
    os << '-';
  }

  os << "\n  Capacity = " << inv.capacity;
  size_type size;
  CUDA_CHECK(cudaMemcpy(&size, inv.d_size_, sizeof(size_type),
                        cudaMemcpyDeviceToHost));
  os << "\n      Size = " << size;
  size_type num_reclaimed;
  CUDA_CHECK(cudaMemcpy(&num_reclaimed, inv.d_num_reclaimed_, sizeof(size_type),
                        cudaMemcpyDeviceToHost));
  os << "\nTombstones = " << num_reclaimed;
  os << "\n     Slots = ";
  for (size_type i{}; i != inv.capacity; ++i) {
    key_type key;
    CUDA_CHECK(cudaMemcpy(&key, &inv.d_key_slots_[i], sizeof(key_type),
                          cudaMemcpyDeviceToHost));
    size_type cnt;
    CUDA_CHECK(cudaMemcpy(&cnt, &inv.d_ref_slots_[i], sizeof(size_type),
                          cudaMemcpyDeviceToHost));

    os << (i ? ",  " : "\n");
    if (i && i % 5 == 0) {
      os << '\n';
    }
    os << '[' << std::setw(3) << i << "] ";

    switch (key) {
      case EMPTY_KEY:
        os << std::setw(9) << (cnt ? "E_ERROR" : "empty");
        break;
      case RECLAIM_KEY:
        os << std::setw(9) << (cnt ? "T_ERROR" : "tombstone");
        break;
      default:
        os << std::setw(3) << key << " = " << std::setw(3) << cnt;
        break;
    }
  }

  os << '\n';
  for (size_type i{}; i < 80; ++i) {
    os << '-';
  }

  return os;
}

/*
template <class Table>
class PrefetchCache {
 public:
  using table_type = Table;

  using value_type = table_type::value_type;
  using meta_type = table_type::meta_type;
  using size_type = table_type::size_type;

  PrefetchCache() = delete;

  PrefetchCache(table_type& dst_table, const size_type capacity,
                cudaStream_t stream = 0);

  ~PrefetchCache();

  void submit_query(const key_type* const d_keys, const size_type num_keys,
                    cudaStream_t stream);

 public:
  // static constexpr uint32_t default_block_size{1024};

  const table_type* const table;
  const size_type table_dims;
  const size_type capacity;

 private:
  size_type inventory_capacity_;
  size_type inventory_size_;
  access_control_type* inventory_access_;
  key_type* inventory_keys_;
  ref_count_type* inventory_counts_;

  value_type* quer_ value_type* values_;
};

template <class K, class R>
__global__ void pc_init_kernel(K* const __restrict keys,
                               R* const __restrict ref_counts const size_t n) {
  const size_t grid_stride{static_cast<size_t>(blockDim.x) * gridDim.x};

  size_t tid{static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
  for (; tid < n; tid += grid_stride) {
    keys[tid] = EMPTY_KEY;
    ref_counts[tid].store(0, cuda::memory_order::relelaxed);
  }
}

template <class Table>
PrefetchCache<Table>::PrefetchCache(table_type& dst_table,
                                    const size_type capacity,
                                    cudaStream_t stream)
    : table{&table}, table_dims{table.dims()}, capacity{capacity} {
  CUDA_CHECK(cudaMallocAsync(&access_, capacity_ * sizeof(access_control_type),
                             stream));
  CUDA_CHECK(cudaMemsetAsync(access_, 0,
                             capacity_ * sizeof(access_control_type), stream));
  CUDA_CHECK(cudaMallocAsync(&keys_, capacity_ * sizeof(key_type), stream));
  CUDA_CHECK(cudaMallocAsync(&ref_counts_, capacity_ * sizeof(ref_count_type),
                             stream));
  CUDA_CHECK(cudaMallocAsync(
      &values_, capacity_ * table_dims * sizeof(value_type), stream));

  const uint32_t block_size{default_block_size};
  const size_t n = capacity;
  const uint32_t grid_size = grid_size_for(n, block_size);
  pc_init_kernel << block_size, grid_size, 0,
      stream >>> (keys_, ref_counts_, n);
}

template <class Table>
PrefetchCache<Table>::~PrefetchCache() {
  CUDA_CHECK(cudaFree(access_));
  CUDA_CHECK(cudaFree(keys_));
  CUDA_CHECK(cudaFree(ref_counts_));
  CUDA_CHECK(cudaFree(values_));
}

__global__ void pc_insert_kernel(const key_type* const keys,
                                 const uint32_t num_keys, const) {
  const uint32_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < num_keys) {
    keys[] if (tid == 0) {}
  }
}

template <class Table>
void PrefetchCache<Table>::submit_query(const key_type* const d_keys,
                                        const size_type num_keys,
                                        cudaStream_t stream) {
  // Eliminate duplicate keys.
  thrust::sort(thrust::device.on(stream), d_keys, d_keys + num_keys);
  thrust::

      const size_t grid_stride{static_cast<size_t>(blockDim.x) * gridDim.x};

  size_t tid{static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
  for (; tid < num_keys; tid += grid_stride) {
    const size_type slot_idx { tid % }

    while (true) {
      keys_[]
    }

    keys_[tid]
  }
}
*/

}  // namespace merlin
}  // namespace nv