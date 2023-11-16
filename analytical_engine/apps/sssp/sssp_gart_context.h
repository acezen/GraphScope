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

#ifndef ANALYTICAL_ENGINE_APPS_PAGERANK_SSSP_GART_CONTEXT_H_
#define ANALYTICAL_ENGINE_APPS_PAGERANK_SSSP_GART_CONTEXT_H_

#include <iomanip>

#include "grape/grape.h"

namespace gs {
/**
 * @brief Context for the Networkx version of PageRank.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class SSSPGartContext
    : public grape::VertexDataContext<FRAG_T, double> {
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;

 public:
  explicit SSSPGartContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, double>(fragment, true),
        partial_result(this->data()) {}

  void Init(grape::DefaultMessageManager& messages, oid_t source_id) {
    LOG(INFO) << "Init context.";
    auto& frag = this->fragment();
    // auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    this->source_id = source_id;
    partial_result.SetValue(std::numeric_limits<double>::max());
    modified.Init(vertices, false);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (partial_result[v] == std::numeric_limits<double>::max()) {
        os << frag.GetId(v) << " "
          << partial_result[v] << std::endl;
      } else {
        os << frag.GetId(v) << " "
          << partial_result[v] << std::endl;
      }
      ++iter;
    }
  }

  oid_t source_id;
  typename FRAG_T::template vertex_array_t<double>& partial_result;
  typename FRAG_T::template vertex_array_t<bool> modified;
  int step;

};
}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_CONTEXT_H_
