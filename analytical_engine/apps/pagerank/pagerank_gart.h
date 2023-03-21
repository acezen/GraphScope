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

#include "apps/pagerank/pagerank_gart_context.h"

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
class PageRankGart
    : public grape::ParallelAppBase<FRAG_T, PageRankGartContext<FRAG_T>>,
      public grape::Communicator,
      public grape::ParallelEngine {
 public:
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kSyncOnOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  INSTALL_PARALLEL_WORKER(PageRankGart<FRAG_T>,
                          PageRankGartContext<FRAG_T>, FRAG_T)

  using vertex_t = typename fragment_t::vertex_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using oid_t = typename fragment_t::oid_t;
  PageRankGart() {}
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    LOG(INFO) << "PEval";
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    Sum(frag.GetInnerVerticesNum(), ctx.graph_vnum);
    LOG(INFO) << "graph_vnum: " << ctx.graph_vnum;
    messages.InitChannels(thread_num());

    ctx.step = 0;
    double p = 1.0 / ctx.graph_vnum;

    // assign initial ranks
    {
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        ctx.pre_result[u] = p;
        ctx.degree[u] =
            static_cast<double>(frag.GetLocalOutDegree(u));
        if (ctx.degree[u] != 0.0) {
          ctx.pre_result[u] = ctx.result[u] / ctx.degree[u];

        // messages.SendMsgThroughIEdges<fragment_t, double>(
        //     frag, u, ctx.result[u] / ctx.degree[u]);
        } else {
          ++ctx.dangling_vnum;
        }
        ++iter;
      }
    }

    {
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        auto es = frag.GetOutgoingAdjList(u);
        auto e_iter = es.begin();
        while (!e_iter.is_end()) {
          ctx.result[e_iter.get_neighbor()] += ctx.pre_result[u];
          ++e_iter;
        }
        ++iter;
      }
    }
    {
      auto iter = outer_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        messages.SyncStateOnOuterVertex(frag, u, ctx.result[u]);
        ++iter;
      }
    }

    double dangling_sum =
        ctx.alpha * p * static_cast<double>(ctx.dangling_vnum);

    Sum(dangling_sum, ctx.dangling_sum);

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    LOG(INFO) << "IncEval";
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    double dangling_sum = ctx.dangling_sum;

    ++ctx.step;
    // process received ranks sent by other workers
    {
      messages.ParallelProcess<fragment_t, double>(
          thread_num(), frag, [&ctx](int tid, const vertex_t& u, const double& msg) {
            ctx.result[u] += msg;
          });
    }

    {
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        ctx.pre_result[u] = ctx.result[u];
        ++iter;
      }
    }

    {
      double base = (1.0 - ctx.alpha) / ctx.graph_vnum + dangling_sum / ctx.graph_vnum;
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        ctx.result[u] = ctx.result[u] * ctx.alpha + base;
        ++iter;
      }
    }

    {
      double eps = 0.0;
      ctx.dangling_sum = 0.0;
      auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;
      if (ctx.degree[u] > 0.0) {
        eps += fabs(ctx.result[u] - ctx.pre_result[u] * ctx.degree[u]);
      } else {
        eps += fabs(ctx.result[u] - ctx.pre_result[u]);
        ctx.dangling_sum += ctx.result[u];
      }
      ++iter;
    }

    double total_eps = 0.0;
    Sum(eps, total_eps);
    if (total_eps < ctx.tolerance * ctx.graph_vnum || ctx.step > ctx.max_round) {
      return;
    }
    }

    {
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;
      ctx.pre_result[u] = ctx.result[u] / ctx.degree[u];
      ++iter;
    }
    }

    ctx.result.SetValue(0.0);

    {
    auto iter = inner_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;
      auto es = frag.GetOutgoingAdjList(u);
      auto e_iter = es.begin();
      while (!e_iter.is_end()) {
        ctx.result[e_iter.get_neighbor()] += ctx.pre_result[u];
        ++e_iter;
      }
      ++iter;
    }
    }

    {
    auto iter = outer_vertices.begin();
    while (!iter.is_end()) {
      auto u = *iter;
      messages.SyncStateOnOuterVertex(frag, u, ctx.result[u]);
      ++iter;
    }
    }

    double new_dangling = ctx.alpha * static_cast<double>(ctx.dangling_sum);
    Sum(new_dangling, ctx.dangling_sum);

    messages.ForceContinue();
  }
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_H_
