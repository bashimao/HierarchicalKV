# [NVIDIA HierarchicalKV(Beta)](https://github.com/NVIDIA-Merlin/HierarchicalKV)

[![Version](https://img.shields.io/github/v/release/NVIDIA-Merlin/HierarchicalKV?color=orange&include_prereleases)](https://github.com/NVIDIA-Merlin/HierarchicalKV/releases)
[![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/HierarchicalKV)](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/HierarchicalKV/master/README.html)

## About HierarchicalKV

HierarchicalKV is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements.

The key capability of HierarchicalKV is to store key-value (feature-embedding) on high-bandwidth memory (HBM) of GPUs and in host memory.

You can also use the library for generic key-value storage.

## Benefits

When building large recommender systems, machine learning (ML) engineers face the following challenges:

- GPUs are needed, but HBM on a single GPU is too small for the large DLRMs that scale to several terabytes.
- Improving communication performance is getting more difficult in larger and larger CPU clusters.
- It is difficult to efficiently control consumption growth of limited HBM with customized strategies.
- Most generic key-value libraries provide low HBM and host memory utilization.

HierarchicalKV alleviates these challenges and helps the machine learning engineers in RecSys with the following benefits:

- Supports training large RecSys models on **HBM and host memory** at the same time.
- Provides better performance by **full bypassing CPUs** and reducing the communication workload.
- Implements table-size restraint strategies that are based on **LRU or customized strategies**.
  The strategies are implemented by CUDA kernels.
- Operates at a high working-status load factor that is close to 1.0.


## Key ideas

- Buckets are locally ordered
- Store keys and values separately
- Store all the keys in HBM
- Build-in and customizable eviction strategy

HierarchicalKV makes NVIDIA GPUs more suitable for training large and super-large models of ***search, recommendations, and advertising***.
The library simplifies the common challenges to building, evaluating, and serving sophisticated recommenders models.

## API Documentation

The main classes and structs are below, but reading the comments in the source code is recommended:

- [`class HashTable`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L151)
- [`class EvictStrategy`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L52)
- [`struct HashTableOptions`](https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/include/merlin_hashtable.cuh#L60)

For regular API doc, please refer to [API Docs](https://nvidia-merlin.github.io/HierarchicalKV/master/api/index.html)

## API Maturity Matrix

`industry-validated` means the API has been well-tested and verified in at least one real-world scenario.

| Name                 | Description                                                                                                           | Function           |
|:---------------------|:----------------------------------------------------------------------------------------------------------------------|:-------------------|
| __insert_or_assign__ | Insert or assign for the specified keys. If the target bucket is full, overwrite the key with minimum score in it.    | industry-validated |
| __insert_and_evict__ | Insert new keys. If the target bucket is full, the keys with minimum score will be evicted for placement the new key. | industry-validated |
| __find_or_insert__   | Search for the specified keys. If missing, insert it.                                                                 | well-tested        |
| __assign__           | Update for each key and ignore the missed one.                                                                        | well-tested        |
| __accum_or_assign__  | Search and update for each key. If found, add value as a delta to the old value. If missing, update it directly.      | well-tested        |
| __find_or_insert\*__ | Search for the specified keys and return the pointers of values. If missing, insert it.                               | well-tested        |
| __find__             | Search for the specified keys.                                                                                        | industry-validated |
| __find\*__           | Search and return the pointers of values, thread-unsafe but with high performance.                                    | well-tested        |
| __export_batch__     | Exports a certain number of the key-value-score tuples.                                                               | industry-validated |
| __export_batch_if__  | Exports a certain number of the key-value-score tuples which match specific conditions.                               | industry-validated |
| __warmup__           | Move the hot key-values from HMEM to HBM                                                                              | June 15, 2023      |

## Usage restrictions

- The `key_type` must be `uint64_t` or `int64_t`.
- The `score_type` must be `uint64_t`.
- The keys of `0xFFFFFFFFFFFFFFFC`, `0xFFFFFFFFFFFFFFFD`, `0xFFFFFFFFFFFFFFFE`, and `0xFFFFFFFFFFFFFFFF` are reserved for internal using.

## Contributors

HierarchicalKV is co-maintianed by [NVIDIA Merlin Team](https://github.com/NVIDIA-Merlin) and NVIDIA product end-users,
and also open for public contributions, bug fixes, and documentation. [[Contribute](CONTRIBUTING.md)]

## How to build

Basically, HierarchicalKV is a headers only library, the commands below only create binaries for benchmark and unit testing.

### with cmake
```shell
git clone --recursive https://github.com/NVIDIA-Merlin/HierarchicalKV.git
cd HierarchicalKV && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -Dsm=80 .. && make -j
```

For Benchmark:
```shell
./merlin_hashtable_benchmark
```

For Unit Test:
```shell
./merlin_hashtable_test
```

### with bazel
```shell
git clone --recursive https://github.com/NVIDIA-Merlin/HierarchicalKV.git
cd HierarchicalKV && bazel build --config=cuda //...
```

For Benchmark:
```shell
./benchmark_util
```

Your environment must meet the following requirements:

- CUDA version >= 11.2
- NVIDIA GPU with compute capability 8.0, 8.6, 8.7 or 9.0

## Benchmark & Performance(W.I.P)

* GPU: 1 x NVIDIA A100 80GB PCIe: 8.0
* Key Type = uint64_t
* Value Type = float32 * {dim}
* Key-Values per OP = 1048576
* Evict strategy: LRU
* `λ`: load factor
* `find*` means the `find` API that directly returns the addresses of values.
* `find_or_insert*` means the `find_or_insert` API that directly returns the addresses of values.
* ***Throughput Unit: Billion-KV/second***

### On pure HBM mode: 

* dim = 8, capacity = 128 Million-KV, HBM = 4 GB, HMEM = 0 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* | insert_and_evict |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|-----------------:|
| 0.50 |            1.151 |  2.535 |          1.691 |  1.927 |  4.127 |           1.799 |            1.071 |
| 0.75 |            0.999 |  2.538 |          0.668 |  0.864 |  1.927 |           1.293 |            1.026 |
| 1.00 |            0.364 |  2.559 |          0.370 |  0.498 |  0.929 |           0.392 |            0.505 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        2.186 |          18.059 |
| 0.75 |        2.163 |          16.762 |
| 1.00 |        2.059 |           2.758 |

* dim = 32, capacity = 128 Million-KV, HBM = 16 GB, HMEM = 0 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* | insert_and_evict |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|-----------------:|
| 0.50 |            1.073 |  2.341 |          1.265 |  1.594 |  4.121 |           1.795 |            0.931 |
| 0.75 |            0.862 |  2.295 |          0.639 |  0.852 |  1.925 |           1.292 |            0.874 |
| 1.00 |            0.359 |  2.345 |          0.345 |  0.492 |  0.926 |           0.375 |            0.467 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.698 |          14.380 |
| 0.75 |        0.574 |          13.489 |
| 1.00 |        0.563 |           0.761 |

* dim = 64, capacity = 64 Million-KV, HBM = 16 GB, HMEM = 0 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* | insert_and_evict |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|-----------------:|
| 0.50 |            0.864 |  2.040 |          0.921 |  1.112 |  4.395 |           1.825 |            0.806 |
| 0.75 |            0.668 |  2.011 |          0.571 |  0.789 |  1.974 |           1.295 |            0.764 |
| 1.00 |            0.333 |  2.050 |          0.334 |  0.469 |  0.938 |           0.392 |            0.481 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.319 |          10.553 |
| 0.75 |        0.298 |          10.400 |
| 1.00 |        0.293 |           0.390 |

### On HBM+HMEM hybrid mode: 

* dim = 64, capacity = 128 Million-KV, HBM = 16 GB, HMEM = 16 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|
| 0.50 |            0.083 |  0.124 |          0.116 |  0.132 |  4.032 |           1.791 |
| 0.75 |            0.082 |  0.123 |          0.114 |  0.129 |  1.906 |           1.131 |
| 1.00 |            0.069 |  0.110 |          0.087 |  0.105 |  0.926 |           0.392 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.319 |          10.789 |
| 0.75 |        0.299 |          10.222 |
| 1.00 |        0.294 |           0.389 |

* dim = 64, capacity = 512 Million-KV, HBM = 32 GB, HMEM = 96 GB

|    λ | insert_or_assign |   find | find_or_insert | assign |  find* | find_or_insert* |
|-----:|-----------------:|-------:|---------------:|-------:|-------:|----------------:|
| 0.50 |            0.049 |  0.073 |          0.049 |  0.070 |  3.535 |           1.718 |
| 0.75 |            0.049 |  0.072 |          0.048 |  0.069 |  1.850 |           1.247 |
| 1.00 |            0.044 |  0.068 |          0.044 |  0.061 |  0.911 |           0.390 |

|    λ | export_batch | export_batch_if |
|-----:|-------------:|----------------:|
| 0.50 |        0.318 |          10.987 |
| 0.75 |        0.298 |          11.213 |
| 1.00 |        0.294 |           0.388 |

### Support and Feedback:

If you encounter any issues or have questions, go to [https://github.com/NVIDIA-Merlin/HierarchicalKV/issues](https://github.com/NVIDIA-Merlin/HierarchicalKV/issues) and submit an issue so that we can provide you with the necessary resolutions and answers.

### Acknowledgment
We are very grateful to external initial contributors [@Zhangyafei](https://github.com/zhangyafeikimi) and [@Lifan](https://github.com/Lifann) for their design, coding, and review work.

### License
Apache License 2.0
