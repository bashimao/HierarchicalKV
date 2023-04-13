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

template <class DataSource>
class BatchQueue {
 public:
  using data_source_type = DataSource;
  using key_type = typename data_source_type::key_type;
  using size_type = size_t;

  BatchQueue(const size_t max_batch_size, const size_t max_queue_length, const size_t num_epochs)
    : max_batch_size_{max_batch_size}, num_epochs_{num_epochs} {
    MERLIN_CHECK(max_batch_size > 0, "max batch size must be > 0.");
    MERLIN_CHECK(max_queue_length > 0, "queue length must be > 0.");
    MERLIN_CHECK(num_epochs > 0, "Must have at least one epoch.")

    // Buffers for holding the batch data.
    CUDA_CHECK(cudaHostAlloc(&h_next_batch_, max_batch_size * sizeof(key_type), cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CUDA_CHECK(cudaMalloc(&d_batches_, max_queue_length * max_batch_size * sizeof(key_type)));
    batch_sizes_.resize(max_queue_length);
  }

  ~BatchQueue() {
    CUDA_CHECK(cudaHostFree(h_next_batch_));
    CUDA_CHECK(cudaFree(d_batches_));
  }

  // Prefetcher API.
  size_type produce_fetch(const key_type** d_batch, cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(modified_mutex_);
    modified_.wait(look, []{ return prod_idx_ - cons_idx_ < batch_sizes_.size(); });

    const size_t i{prod_idx_ % batch_sizes_.size()};  

    const size_t batch_size{data_source_.fetch_next(max_batch_size_, h_next_batch_)};
    batch_sizes_[i] = batch_size;
    
    d_batch = &d_batches_[i * max_batch_size_];
    CUDA_CHECK(cudaMemcpyAsync(d_batch, sizeof(key_type) * batch_size, cudaMemcpyHostToDevice, stream));
    
    return batch_size;
  }
  void produce_commit(const key_type* const d_batch, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t i{prod_idx_++ % batch_sizes_.size()};
    assert(d_batches_[i * max_batch_size_] == d_batch);

    modified_.notify_one();
  }

  // Trainer API.
  size_type consume_fetch(const key_type** d_batch) {
    std::unique_lock<std::mutex> lock(modified_mutex_);
    modified_.wait(look, []{ return prod_idx_ > cons_idx_; });

    const size_t i{cons_idx_ % batch_sizes_.size()};
    d_batch = &d_batches_[i * max_batch_size_];
    return batch_sizes_[i];
  }
  void consume_commit(const key_type* const d_batch, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    const size_t i{cons_idx_++ % batch_sizes_.size()};
    assert(d_batches_[i * max_batch_size_] == d_batch);

    modified_.notify_one();
  }

 private:
  const size_t max_batch_size_;

  data_source_type data_source_;
  key_type* h_next_batch_;
  
  key_type* d_batches_;
  std::vector<size_t> batch_sizes_;

  size_t prod_idx_{};
  size_t cons_idx_{};

  std::mutex modified_mutex_;
  std::condition_variable modified_;
};