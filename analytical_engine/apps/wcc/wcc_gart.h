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

#ifndef ANALYTICAL_ENGINE_APPS_PAGERANK_WCC_GART_H_
#define ANALYTICAL_ENGINE_APPS_PAGERANK_WCC_GART_H_

#include <queue>
#include <utility>
#include <vector>
#include <iostream>

#include "grape/grape.h"

#include "core/app/app_base.h"
#include "core/utils/trait_utils.h"

namespace gs {

template <typename FRAG_T>
class WCCGartContext
    : public grape::VertexDataContext<FRAG_T, typename FRAG_T::vid_t> {
  using vid_t = typename FRAG_T::vid_t;

 public:
  explicit WCCGartContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, typename FRAG_T::vid_t>(fragment,
                                                                 true),
        comp_id(this->data()) {}

  void Init(grape::DefaultMessageManager& messages) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    curr_modified.Init(vertices, false);
    next_modified.Init(vertices, false);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      os << frag.GetId(v) << " " << comp_id[v] << std::endl;
      ++iter;
    }
  }

  typename FRAG_T::template vertex_array_t<vid_t>& comp_id;
  typename FRAG_T::template vertex_array_t<bool> curr_modified;
  typename FRAG_T::template vertex_array_t<bool> next_modified;
  int step = 0;
};


template <typename FRAG_T>
class WCCGart
    : public AppBase<FRAG_T, WCCGartContext<FRAG_T>> {
 public:
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kSyncOnOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  INSTALL_DEFAULT_WORKER(WCCGart<FRAG_T>,
                         WCCGartContext<FRAG_T>, FRAG_T)

  using vertex_t = typename fragment_t::vertex_t;
  using vid_t = typename fragment_t::vid_t;

  WCCGart() {}

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    LOG(INFO) << "PEval";

    ctx.step = 0;

    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      ctx.comp_id[v] = frag.GetInnerVertexGid(v);
      ++iter;
    }

    iter = outer_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      ctx.comp_id[v] = frag.GetInnerVertexGid(v);
      ++iter;
    }

    // compute
    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      auto cid = ctx.comp_id[v];

      auto es = frag.GetOutgoingAdjList(v);
      auto e_iter = es.begin();
      while (!e_iter.is_end()) {
        auto u = e_iter.get_neighbor();
        if (ctx.comp_id[u] > cid) {
          ctx.comp_id[u] = cid;
          ctx.next_modified[u] = true;
        }
        ++e_iter;
      }

      auto es2 = frag.GetIncomingAdjList(v);
      if (frag.directed()) {
        e_iter = es2.begin();
        while (!e_iter.is_end()) {
          auto u = e_iter.get_neighbor();
          if (ctx.comp_id[u] > cid) {
            ctx.comp_id[u] = cid;
            ctx.next_modified[u] = true;
          }
          ++e_iter;
        }
      }
      ++iter;
    }

    // send message
    iter = outer_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.next_modified[v]) {
        messages.SyncStateOnOuterVertex<FRAG_T, vid_t>(frag, v, ctx.comp_id[v]);
        ctx.next_modified[v] = false;
      }
      ++iter;
    }

    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.next_modified[v]) {
        messages.ForceContinue();
        break;
      }
      ++iter;
    }

    ctx.next_modified.Swap(ctx.curr_modified);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    ctx.step++;
    LOG(INFO) << "IncEval-" << ctx.step;

		{
      vertex_t v(0);
      vid_t val;
      while (messages.GetMessage<fragment_t, vid_t>(frag, v, val)) {
        if (ctx.comp_id[v] > val) {
          ctx.comp_id[v] = val;
          ctx.curr_modified[v] = true;
        }
      }
    }

    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    // compute
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;

      if (!ctx.curr_modified[v]) {
        ++iter;
        continue;
      }

      ctx.curr_modified[v] = false;
      auto cid = ctx.comp_id[v];

      auto es = frag.GetOutgoingAdjList(v);
      auto e_iter = es.begin();
      while (!e_iter.is_end()) {
        auto u = e_iter.get_neighbor();
        if (ctx.comp_id[u] > cid) {
          ctx.comp_id[u] = cid;
          ctx.next_modified[u] = true;
        }
        ++e_iter;
      }

      auto es2 = frag.GetIncomingAdjList(v);
      if (frag.directed()) {
        e_iter = es2.begin();
        while (!e_iter.is_end()) {
          auto u = e_iter.get_neighbor();
          if (ctx.comp_id[u] > cid) {
            ctx.comp_id[u] = cid;
            ctx.next_modified[u] = true;
          }
          ++e_iter;
        }
      }

      ++iter;
    }

    // send message
    iter = outer_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.next_modified[v]) {
        messages.SyncStateOnOuterVertex<FRAG_T, vid_t>(frag, v, ctx.comp_id[v]);
        ctx.next_modified[v] = false;
      }
      ++iter;
    }

    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.next_modified[v]) {
        messages.ForceContinue();
        break;
      }
      ++iter;
    }

    ctx.next_modified.Swap(ctx.curr_modified);
  }
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_WCC_GART_H_
