/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ANALYTICAL_ENGINE_APPS_SEAL_SEAL_CONTEXT_H_
#define ANALYTICAL_ENGINE_APPS_SEAL_SEAL_CONTEXT_H_

#include <limits>
#include <queue>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"

#include "grape/grape.h"

#include "core/context/tensor_context.h"

using json = nlohmann::json;

namespace gs {

template <typename FRAG_T>
class SealPathContext : public TensorContext<FRAG_T, typename FRAG_T::oid_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using path_t = std::vector<vid_t>;

  explicit SealPathContext(const FRAG_T& fragment)
      : TensorContext<FRAG_T, typename FRAG_T::oid_t>(fragment) {}

  void Init(grape::DefaultMessageManager& messages, std::string pairs,
            int k, int n) {
    auto& frag = this->fragment();
    auto vm_ptr = frag.GetVertexMap();
    auto fid = frag.fid();
    vid_t src, dst;

    auto pairs_json = json::parse(pairs.c_str());
    if (!pairs_json.empty()) {
      path_queues.resize(pairs_json.size());
      for (size_t i = 0; i < pairs_json.size(); ++i) {
      for (auto& pair : pairs_json) {
        if (pair[i].is_array() && pair[i].size() == 2) {
          if (vm_ptr->GetGid(fid, pair[i][0].get<oid_t>(), src) &&
              vm_ptr->GetGid(pair[i][1].get<oid_t>(), dst)) {
            auto new_pair = std::make_pair(dst, path_t({src}));
            this->path_queues[i].push(new_pair);
          }
        } else {
          LOG(ERROR) << "Invalid pair: " << pair.dump();
        }
      }
    }

    this->k = k;
    this->n = n;

#ifdef PROFILING
    preprocess_time = 0;
    exec_time = 0;
    postprocess_time = 0;
#endif
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();

    for (auto& path : path_result) {
      std::string buf;

      for (auto gid : path) {
        buf += std::to_string(frag.Gid2Oid(gid)) + " ";
      }
      if (!buf.empty()) {
        buf[buf.size() - 1] = '\n';
        os << buf;
      }
    }

#ifdef PROFILING
    VLOG(2) << "preprocess_time: " << preprocess_time << "s.";
    VLOG(2) << "exec_time: " << exec_time << "s.";
    VLOG(2) << "postprocess_time: " << postprocess_time << "s.";
#endif
  }

  // std::vector<std::pair<vid_t, vid_t>> pairs;
  std::vector<std::queue<std::pair<vid_t, path_t>>> path_queues;
  int k, n;
  std::vector<std::vector<path_t>> path_results;

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif

};
}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_SEAL_SEAL_CONTEXT_H_
