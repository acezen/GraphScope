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

#ifndef ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_GART_H_
#define ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_GART_H_

#include "grape/grape.h"

#include "core/app/app_base.h"
#include "apps/pagerank/pagerank_gart_context.h"

namespace gs {

template <typename FRAG_T>
class PageRankGart
    : public AppBase<FRAG_T, PageRankGartContext<FRAG_T>>,
      public grape::Communicator {
 public:
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  INSTALL_DEFAULT_WORKER(PageRankGart<FRAG_T>,
                         PageRankGartContext<FRAG_T>, FRAG_T)

  using vertex_t = typename fragment_t::vertex_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using fid_t = typename fragment_t::fid_t;

  PageRankGart() {}

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    LOG_IF(INFO, frag.fid() == 0) << "PEval";

    auto inner_vertices = frag.InnerVertices();

    Sum(frag.GetInnerVerticesNum(), ctx.graph_vnum);

    ctx.step = 0;
    double p = 1.0 / ctx.graph_vnum;

    // assign initial ranks
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;

      ctx.result[u] = p;
      ctx.degree[u] =
        static_cast<double>(frag.GetLocalOutDegree(u));

      if (ctx.degree[u] != 0.0) {
        messages.SendMsgThroughOEdges<fragment_t, double>(
            frag, u, ctx.result[u] / ctx.degree[u]);
      }
      ++iter;
    }

    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;

      if (ctx.degree[u] == 0.0) {
        ++ctx.dangling_vnum;
      }
       ++iter;
    }

    double dangling_sum =
        ctx.alpha * p * static_cast<double>(ctx.dangling_vnum);

    Sum(dangling_sum, ctx.dangling_sum);

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {

    ctx.step++;
    LOG_IF(INFO, frag.fid() == 0) << "IncEval-" << ctx.step;

    auto inner_vertices = frag.InnerVertices();

    double dangling_sum = ctx.dangling_sum;

    double t = grape::GetCurrentTime();
    // process received ranks sent by other workers
    {
      vertex_t v(0);
      double val;

      while (messages.GetMessage<fragment_t, double>(frag, v, val)) {
        ctx.result[v] = val;
        ctx.pre_result[v] = val;
      }
    }

    LOG_IF(INFO, frag.fid() == 0) << "Message process: " << grape::GetCurrentTime() - t << " seconds";

    t = grape::GetCurrentTime();
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;

      if (ctx.degree[u] > 0.0) {
        ctx.pre_result[u] = ctx.result[u] / ctx.degree[u];
      } else {
        ctx.pre_result[u] = ctx.result[u];
      }
      ++iter;
    }
    LOG_IF(INFO, frag.fid() == 0) << "Process pre_result: " << grape::GetCurrentTime() - t << " seconds";

    t = grape::GetCurrentTime();
    double base = (1.0 - ctx.alpha) / ctx.graph_vnum + dangling_sum / ctx.graph_vnum;
    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;

      double cur = 0;
      if (frag.directed()) {
        auto es = frag.GetIncomingAdjList(u);
        auto e_iter = es.begin();
        while (!e_iter.is_end()) {
          cur += ctx.pre_result[e_iter.get_neighbor()];
          ++e_iter;
        }
      } else {
        auto es = frag.GetOutgoingAdjList(u);
        auto e_iter = es.begin();
        while (!e_iter.is_end()) {
           cur += ctx.pre_result[e_iter.get_neighbor()];
           ++e_iter;
        }
      }
      ctx.result[u] = cur * ctx.alpha + base;
      ++iter;
    }
    LOG_IF(INFO, frag.fid() == 0) << "Compute: " << grape::GetCurrentTime() - t << " seconds";

    t = grape::GetCurrentTime();
    double eps = 0.0;
    ctx.dangling_sum = 0.0;
    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto v = *iter;
      if (ctx.degree[v] > 0.0) {
        eps += fabs(ctx.result[v] - ctx.pre_result[v] * ctx.degree[v]);
      } else {
        eps += fabs(ctx.result[v] - ctx.pre_result[v]);
        ctx.dangling_sum += ctx.result[v];
      }
      ++iter;
    }
    double total_eps = 0.0;
    Sum(eps, total_eps);
    if (total_eps < ctx.tolerance * ctx.graph_vnum || ctx.step > ctx.max_round) {
      return;
    }

    iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;
      if (ctx.degree[u] > 0) {
        messages.SendMsgThroughOEdges<fragment_t, double>(
            frag, u, ctx.result[u] / ctx.degree[u]);
      }
      ++iter;
    }
    LOG_IF(INFO, frag.fid() == 0) << "Update: " << grape::GetCurrentTime() - t << " seconds";

    double new_dangling = ctx.alpha * static_cast<double>(ctx.dangling_sum);
    Sum(new_dangling, ctx.dangling_sum);

    messages.ForceContinue();
  }
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_GART_H_
