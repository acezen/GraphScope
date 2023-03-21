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

#ifndef ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_CONTEXT_H_
#define ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_CONTEXT_H_

#include <iomanip>

#include "grape/grape.h"

namespace gs {
/**
 * @brief Context for the Networkx version of PageRank.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PageRankNetworkXContext
    : public grape::VertexDataContext<FRAG_T, double> {
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;

 public:
  explicit PageRankNetworkXContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, double>(fragment, true),
        result(this->data()) {}

  void Init(grape::ParallelMessageManager& messages, double alpha,
            int max_round, double tolerance) {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    /*
    double t = grape::GetCurrentTime();
    for (const auto& v : inner_vertices) {
      ;
    }
    grape::Vertex v(0);
    auto it = inner_vertices.begin_;
    auto end = inner_vertices.end_;
    while (it != end) {
      grin_get_vertex_from_opt(static_cast<GRIN_GRAPH>(&v), inner_vertices.vl_, it);
      // grin_destroy_vertex(inner_vertices.g_, v);
      ++it;
    }
    // auto it = inner_vertices.begin();
    // auto end = inner_vertices.end();
    // while (it != end) {
    //  auto& v = *it;
    //   ++it;
    // }
    LOG_IF(INFO, frag.fid() == 0) << "Iterate vertices: " << grape::GetCurrentTime() - t << " seconds ";

    t = grape::GetCurrentTime();
    auto it = inner_vertices.begin_;
    auto end = inner_vertices.end_;
    grape::Vertex v(it);
    grape::Vertex nv(it);
    auto grinv = static_cast<GRIN_VERTEX>(&nv);
    vertex_t gv(inner_vertices.g_, static_cast<GRIN_VERTEX>(&v));
    vertex_t egv(inner_vertices.g_, static_cast<GRIN_VERTEX>(&nv));
    while (it != end) {
      // grin_get_vertex_from_opt(static_cast<GRIN_GRAPH>(&v), inner_vertices.vl_, it);
      // grin_destroy_vertex(inner_vertices.g_, v);
      auto es = frag.GetIncomingAdjList(gv);
      auto eit = es.begin();
      auto eend = es.end();
      for (auto eit = es.begin(); eit != eend; ++eit) {
        eit->get_neighbor(egv);
        // nei.grin_v = nullptr;
        // e.get_neighbor();
      }
      // for (auto& e : es) {
      //   e.get_neighbor();
      // }
      ++it;
    }
    egv.grin_v = nullptr;
    gv.grin_v = nullptr;
    for (const auto& v : inner_vertices) {
      auto es = frag.GetIncomingAdjList(v);
      for (auto& e : es) {
        auto u = e.get_neighbor();
      }
    }
    LOG_IF(INFO, frag.fid() == 0) << "Iterate Edges: " << grape::GetCurrentTime() - t << " seconds ";
    // Arrow: 0.964705s
    // GRIN: 5.27114s
    // GRIN with IsInnerVertex(true): 1.29798s
    double t = grape::GetCurrentTime();
    size_t count = 0;
    for (const auto& v : inner_vertices) {
      auto es = frag.GetIncomingAdjList(v);
      for (auto& e : es) {
        auto u = e.get_neighbor();
        if (frag.IsInnerVertex(u)) {
          ++count;
        }
      }
    }
    */
    // LOG_IF(INFO, frag.fid() == 0) << "Iterate Edges: " << grape::GetCurrentTime() - t << " seconds " << count;

    this->alpha = alpha;
    this->max_round = max_round;
    this->tolerance = tolerance;
    degree.Init(inner_vertices, 0);
    result.SetValue(0.0);
    pre_result.Init(vertices, 0.0);
    // GRIN: 20.9127s
    // GRIN-get_neighbor_directly: 17.6011s
    // GRIN-adj_list_no_destroy: 15.4245s
    // GRIN-relace all grin function with its implement: 12.8841s
    // GRIN-get location with v.grin_v: 8.5274s
    // ARROW: 7.89582s
    /*
    double t = grape::GetCurrentTime();
    double cur = 0;
    for (const auto& v : inner_vertices) {
      auto es = frag.GetIncomingAdjList(v);
      for (auto& e : es) {
        auto u = e.get_neighbor();
        cur += pre_result[u];
      }
    }
    LOG_IF(INFO, frag.fid() == 0) << "Iterate Edges: " << grape::GetCurrentTime() - t << " seconds " << cur;
    */
    step = 0;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();
    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << result[v] << std::endl;
    }
  }

  typename FRAG_T::template inner_vertex_array_t<double> degree;
  typename FRAG_T::template vertex_array_t<double>& result;
  typename FRAG_T::template vertex_array_t<double> pre_result;

  uint64_t dangling_vnum = 0;
  int step = 0;
  int max_round = 0;
  double alpha = 0;
  double tolerance;

  double dangling_sum = 0.0;
  size_t graph_vnum = 0;
};
}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_CONTEXT_H_
