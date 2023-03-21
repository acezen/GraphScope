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

#ifndef ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_H_
#define ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_H_

#include "grape/grape.h"

#include "apps/pagerank/pagerank_networkx_context.h"

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
class PageRankNetworkX
    : public grape::ParallelAppBase<FRAG_T, PageRankNetworkXContext<FRAG_T>>,
      public grape::Communicator,
      public grape::ParallelEngine {
 public:
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  INSTALL_PARALLEL_WORKER(PageRankNetworkX<FRAG_T>,
                          PageRankNetworkXContext<FRAG_T>, FRAG_T)

  using vertex_t = typename fragment_t::vertex_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  PageRankNetworkX() {}
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    LOG_IF(INFO, frag.fid() == 0) << "PEval";
    auto inner_vertices = frag.InnerVertices();

    Sum(frag.GetInnerVerticesNum(), ctx.graph_vnum);
    messages.InitChannels(thread_num());

    ctx.step = 0;
    double p = 1.0 / ctx.graph_vnum;

    // assign initial ranks
    ForEach(inner_vertices.begin(), inner_vertices.end(),
            [&ctx, &frag, p, &messages](int tid, const vertex_t& u) {
              ctx.result[u] = p;
              ctx.degree[u] =
                  static_cast<double>(frag.GetLocalOutDegree(u));
              if (ctx.degree[u] != 0.0) {
                messages.SendMsgThroughOEdges<fragment_t, double>(
                    frag, u, ctx.result[u] / ctx.degree[u], tid);
              }
            });

    for (auto u : inner_vertices) {
      if (ctx.degree[u] == 0.0) {
        ++ctx.dangling_vnum;
      }
    }

    double dangling_sum =
        ctx.alpha * p * static_cast<double>(ctx.dangling_vnum);

    Sum(dangling_sum, ctx.dangling_sum);

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    LOG_IF(INFO, frag.fid() == 0) << "IncEval";
    auto inner_vertices = frag.InnerVertices();

    double dangling_sum = ctx.dangling_sum;

    ++ctx.step;
    double t = grape::GetCurrentTime();
    // process received ranks sent by other workers
    {
      messages.ParallelProcess<fragment_t, double>(
          thread_num(), frag, [&ctx](int tid, const vertex_t& u, const double& msg) {
            ctx.result[u] = msg;
            ctx.pre_result[u] = msg;
          });
    }
    LOG_IF(INFO, frag.fid() == 0) << "Message process: " << grape::GetCurrentTime() - t << " seconds";

    t = grape::GetCurrentTime();
    ForEach(inner_vertices.begin(), inner_vertices.end(),
            [&ctx](int tid, const vertex_t& u) {
              if (ctx.degree[u] > 0.0) {
                ctx.pre_result[u] = ctx.result[u] / ctx.degree[u];
              } else {
                ctx.pre_result[u] = ctx.result[u];
              }
            });
    LOG_IF(INFO, frag.fid() == 0) << "Process pre_result: " << grape::GetCurrentTime() - t << " seconds";

    t = grape::GetCurrentTime();
    double base = (1.0 - ctx.alpha) / ctx.graph_vnum + dangling_sum / ctx.graph_vnum;
    ForEach(inner_vertices.begin(), inner_vertices.end(),
            [&ctx, base, &frag](int tid, const vertex_t& u) {
              double cur = 0;
              if (frag.directed()) {
                auto es = frag.GetIncomingAdjList(u);
                for (auto& e : es) {
                  cur += ctx.pre_result[e.get_neighbor()];
                }
              } else {
                auto es = frag.GetOutgoingAdjList(u);
                for (auto& e : es) {
                  cur += ctx.pre_result[e.get_neighbor()];
                }
              }
              ctx.result[u] = cur * ctx.alpha + base;
            });

    LOG_IF(INFO, frag.fid() == 0) << "Compute: " << grape::GetCurrentTime() - t << " seconds";
    t = grape::GetCurrentTime();
    double eps = 0.0;
    ctx.dangling_sum = 0.0;
    for (const auto& v : inner_vertices) {
      if (ctx.degree[v] > 0.0) {
        eps += fabs(ctx.result[v] - ctx.pre_result[v] * ctx.degree[v]);
      } else {
        eps += fabs(ctx.result[v] - ctx.pre_result[v]);
        ctx.dangling_sum += ctx.result[v];
      }
    }
    double total_eps = 0.0;
    Sum(eps, total_eps);
    if (total_eps < ctx.tolerance * ctx.graph_vnum || ctx.step > ctx.max_round) {
      return;
    }

    ForEach(inner_vertices.begin(), inner_vertices.end(),
            [&ctx, &frag, &messages](int tid, const vertex_t& u) {
              if (ctx.degree[u] > 0) {
                messages.SendMsgThroughOEdges<fragment_t, double>(
                    frag, u, ctx.result[u] / ctx.degree[u], tid);
              }
            });
    LOG_IF(INFO, frag.fid() == 0) << "Update: " << grape::GetCurrentTime() - t << " seconds";

    double new_dangling = ctx.alpha * static_cast<double>(ctx.dangling_sum);
    Sum(new_dangling, ctx.dangling_sum);

    messages.ForceContinue();
  }
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_H_
