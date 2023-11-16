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

#ifndef ANALYTICAL_ENGINE_APPS_PAGERANK_SSSP_GART_H_
#define ANALYTICAL_ENGINE_APPS_PAGERANK_SSSP_GART_H_

#include <queue>

#include "grape/grape.h"

#include "core/app/app_base.h"
#include "core/utils/trait_utils.h"
#include "apps/sssp/sssp_gart_context.h"

namespace gs {

/**
 * @brief An implementation of PageRank, the version in Networkx, which can work
 * on directed graphs.
 *
 * This version of PageRank inherits ParallelAppBase. Messages can be sent in
 * parallel with the evaluation process. This strategy improves performance by
 * overlapping the communication time and the evaluation time.
 *
 * @tparam FRAG_T
 */

template <typename FRAG_T>
class SSSPGart
    : public AppBase<FRAG_T, SSSPGartContext<FRAG_T>> {
 public:
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kSyncOnOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  INSTALL_DEFAULT_WORKER(SSSPGart<FRAG_T>,
                         SSSPGartContext<FRAG_T>, FRAG_T)

  using vertex_t = typename fragment_t::vertex_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using oid_t = typename fragment_t::oid_t;
  SSSPGart() {}
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    LOG(INFO) << "PEval";

    ctx.step = 0;
    vertex_t source(0);
    // bool native_source = false;
    // if (frag.fid() == 1) {
      // native_source = true;
    // }
    bool native_source = frag.GetVertex(ctx.source_id, source);
    if (native_source && frag.IsInnerVertex(source)) {
      LOG(INFO) << "native source " << frag.fid() << " " << frag.GetId(source);
      native_source = true;
    } else {
      native_source = false;
    }

    std::priority_queue<std::pair<double, vertex_t>> heap;

    if (native_source) {
      /*
      auto iter = frag.InnerVertices().begin();
      while (!iter.is_end()) {
        auto v = *iter;
        if (frag.GetId(v) == 5) {
          LOG(INFO) << "source " << frag.fid() << " " << frag.GetId(v);
          ctx.partial_result[v] = 0.0;
          heap.emplace(0, v);
          break;
        }
        ++iter;
      }
      */
      ctx.partial_result[source] = 0.0;
      heap.emplace(0, source);
    }

    Dijkstra(frag, ctx, heap);

    auto outer_vertices = frag.OuterVertices();
    auto iter = outer_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.modified[v]) {
        messages.SyncStateOnOuterVertex<FRAG_T, double>(frag, v,
                                                        ctx.partial_result[v]);
      }
      ++iter;
    }

    ctx.modified.SetValue(false);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    ctx.step++;
    LOG(INFO) << "IncEval-" << ctx.step;

    auto inner_vertices = frag.InnerVertices();

    std::priority_queue<std::pair<double, vertex_t>> heap;

    {
      vertex_t v(0);
      double val;
      while (messages.GetMessage<fragment_t, double>(frag, v, val)) {
        if (val < ctx.partial_result[v]) {
          ctx.partial_result[v] = val;
          ctx.modified[v] = true;
        }
      }
    }

    {
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.modified[v]) {
        heap.emplace(-ctx.partial_result[v], v);
        ctx.modified[v] = false;
      }
      ++iter;
    }
    }

    Dijkstra(frag, ctx, heap);

    {
    auto outer_vertices = frag.OuterVertices();
    auto iter = outer_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.modified[v]) {
        messages.SyncStateOnOuterVertex<FRAG_T, double>(frag, v,
                                                        ctx.partial_result[v]);
      }
      ++iter;
    }
    }
    ctx.modified.SetValue(false);
  }

 private:
  // sequential Dijkstra algorithm for SSSP.
  void Dijkstra(const fragment_t& frag, context_t& ctx,
                std::priority_queue<std::pair<double, vertex_t>>& heap) {
    double distu, distv, ndistv;
    vertex_t v, u;

    while (!heap.empty()) {
      u = std::move(heap.top().second);
      distu = -heap.top().first;
      heap.pop();

      if (ctx.modified[u]) {
        continue;
      }
      ctx.modified[u] = true;

      auto es = frag.GetOutgoingAdjList(u);
      auto e_iter = es.begin();
      while (!e_iter.is_end()) {
        v = std::move(e_iter.get_neighbor());
        distv = ctx.partial_result[v];
        double edata = 1.0;
        vineyard::static_if<!std::is_same<edata_t, grape::EmptyType>{}>(
            [&](auto& e, auto& data) {
              data = static_cast<double>(e.get_data());
            })(e_iter, edata);
        ndistv = distu + edata;
        if (distv > ndistv) {
          ctx.partial_result[v] = ndistv;
          if (frag.IsInnerVertex(v)) {
            heap.emplace(-ndistv, v);
          } else {
            ctx.modified[v] = true;
          }
        }
        ++e_iter;
      }
    }
  }
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_H_
