/** Copyright 2020 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANALYTICAL_ENGINE_APPS_PROJECTED_SSSP_PROJECTED_H_
#define ANALYTICAL_ENGINE_APPS_PROJECTED_SSSP_PROJECTED_H_

#include <iostream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "grape/grape.h"

#include "core/app/app_base.h"
#include "core/utils/trait_utils.h"
#include "core/worker/default_worker.h"

namespace gs {

template <typename FRAG_T>
class SSSPProjectedContext : public grape::VertexDataContext<FRAG_T, int64_t> {
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;

 public:
  explicit SSSPProjectedContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, int64_t>(fragment, true),
        partial_result(this->data()) {}

  void Init(grape::DefaultMessageManager& messages, oid_t source_id_) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    source_id = source_id_;
    partial_result.SetValue(std::numeric_limits<int64_t>::max());
    modified.Init(vertices, false);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    for (auto v : iv) {
      os << frag.GetId(v) << "\t" << partial_result[v] << std::endl;
    }
  }

  typename FRAG_T::template vertex_array_t<int64_t>& partial_result;
  typename FRAG_T::template vertex_array_t<bool> modified;
  oid_t source_id;
};

template <typename FRAG_T>
class SSSPProjected : public AppBase<FRAG_T, SSSPProjectedContext<FRAG_T>> {
 public:
  // specialize the templated worker.
  INSTALL_DEFAULT_WORKER(SSSPProjected<FRAG_T>, SSSPProjectedContext<FRAG_T>,
                         FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;
  using edata_t = typename fragment_t::edata_t;

 private:
  // sequential Dijkstra algorithm for SSSP.
  void Dijkstra(const fragment_t& frag, context_t& ctx,
                std::priority_queue<std::pair<int64_t, vertex_t>>& heap) {
    // auto inner_vertices = frag.InnerVertices();

    int64_t distu, distv, ndistv;

    while (!heap.empty()) {
      auto u = heap.top().second;
      distu = -heap.top().first;
      heap.pop();

      if (ctx.modified[u]) {
        continue;
      }
      ctx.modified[u] = true;

      auto es = frag.GetOutgoingAdjList(u);
      for (const auto& e : es) {
        auto v = e.get_neighbor();
        distv = ctx.partial_result[v];
        int64_t edata = 1.0;
        vineyard::static_if<!std::is_same<edata_t, grape::EmptyType>{}>(
            [&](auto& e, auto& data) {
              data = static_cast<int64_t>(e.get_data());
            })(e, edata);
        ndistv = distu + edata;
        if (distv > ndistv) {
          ctx.partial_result[v] = ndistv;
          if (frag.IsInnerVertex(v)) {
            heap.emplace(-ndistv, v);
          } else {
            ctx.modified[v] = true;
          }
        }
      }
    }
  }

 public:
  void PEval(const fragment_t& frag, context_t& ctx,
             grape::DefaultMessageManager& messages) {
    LOG_IF(INFO, frag.fid() == 0) << "PEval";
    vertex_t source;
    bool native_source = frag.GetInnerVertex(ctx.source_id, source);

    std::priority_queue<std::pair<int64_t, vertex_t>> heap;

    if (native_source) {
      ctx.partial_result[source] = 0.0;
      heap.emplace(0, source);
    }

    Dijkstra(frag, ctx, heap);

    auto outer_vertices = frag.OuterVertices();
    for (const auto& v : outer_vertices) {
      if (ctx.modified[v]) {
        messages.SyncStateOnOuterVertex<FRAG_T, int64_t>(frag, v,
                                                        ctx.partial_result[v]);
      }
    }

    ctx.modified.SetValue(false);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               grape::DefaultMessageManager& messages) {
    LOG_IF(INFO, frag.fid() == 0) << "IncEval";
    auto inner_vertices = frag.InnerVertices();

    std::priority_queue<std::pair<int64_t, vertex_t>> heap;

    {
      vertex_t v(0);
      int64_t val;
      while (messages.GetMessage<fragment_t, int64_t>(frag, v, val)) {
        if (val < ctx.partial_result[v]) {
          ctx.partial_result[v] = val;
          ctx.modified[v] = true;
        }
      }
    }

    for (const auto& v : inner_vertices) {
      if (ctx.modified[v]) {
        heap.emplace(-ctx.partial_result[v], v);
        ctx.modified[v] = false;
      }
    }

    Dijkstra(frag, ctx, heap);

    auto outer_vertices = frag.OuterVertices();
    for (const auto& v : outer_vertices) {
      if (ctx.modified[v]) {
        messages.SyncStateOnOuterVertex<FRAG_T, int64_t>(frag, v,
                                                        ctx.partial_result[v]);
      }
    }
    ctx.modified.SetValue(false);
  }
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PROJECTED_SSSP_PROJECTED_H_
