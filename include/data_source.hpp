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

#include <random>

#pragma once

namespace nv {
namespace merlin {

/**
 * Spams keys from a uniform_int distribution.
 */
template <class Distribution, class Generator = std::default_random_engine>
class RandomDataSource {
 public:
  using gen_type = Generator;
  using dist_type = Distribution;
  using key_type = typename dist_type::result_type;

  RandomDataSource(const dist_type& dist, const size_t length)
      : dist_{dist}, length_{length} {
    assert(length_ > 0);
  }

  void reset() {
    index_ = 0;
  }

  inline bool has_next() const { return index_ <  length_; }

  size_t fetch_next(size_t max_batch_size, key_type* const batch) {
    assert(max_batch_size > 0 && batch);
    assert(index_ < length_);

    max_batch_size = std::min(max_batch_size, length_ - index_);

    size_t n{};
    while (n != max_batch_size) {
      batch[n++] = dist_(gen_);
    }
    index_ += n;
    return n;
  }

 private:
  gen_type gen_;
  dist_type dist_;
  const size_t length_;
  size_t index_{};
};

template <class K>
class RawFileDataSource {
 public:
  using key_type = K

  RawFileDataSource(const std::vector<std::string>& paths) {
    assert(!paths.empty());

    files_.reserve(paths.size());
    for (const std::string& path : paths) {
      files_.emplace_back(std::fopen(path.c_str(), "r"));
    }
    current_ = files_.begin();
  }

  void reset() {
    for (std::FILE* file : files_) {
      MERLIN_CHECK(std::fseek(file, 0, SEEK_SET) == 0, "Unable to reset data source!");
    }
    current_ = files_.begin();
  }

  inline bool has_next() const { return cur_file_ < files.end() }

  size_t fetch_next(const size_t max_batch_size, key_type* const batch) {
    assert(max_batch_size > 0 && batch);
    assert(cur_file_ < files_.end());

    size_t n{};
    do {
      n += std::fread(&batch[n], sizeof(key_type), max_batch_size - n, *cur_file_);
      if (n == max_batch_size) {
        break;
      }
    } while (++cur_file_ < files_.end());
    return n;
  }

 private:
  std::vector<std::FILE*> files_;
  std::vector<std::FILE*>::iterator cur_file_;
};

}
}