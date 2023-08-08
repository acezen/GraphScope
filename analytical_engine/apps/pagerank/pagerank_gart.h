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
    // LOG(INFO) << "PEval" << " frag = " << frag.fid() << " thread_num() = " << thread_num();
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    if (false) {
      double start = grape::GetCurrentTime();
      for (int idx = 0; idx < 100; idx ++) {
        auto iter = inner_vertices.begin();
        while (!iter.is_end()) {
          auto u = *iter;
          ++iter;
        }
      }
      if (frag.fid() == 0) {
        LOG(INFO) << "PEval stage prepare vertex Query time: " << grape::GetCurrentTime() - start << " seconds";
      }
      return;
    }
    
    if (true) {
      double start = grape::GetCurrentTime();
      auto iter = inner_vertices.begin();
      int total_edge_num = 0;
      while (!iter.is_end()) {
        auto u = *iter;
          auto es = frag.GetOutgoingAdjList(u);
          auto e_iter = es.begin();
          
          while (!e_iter.is_end()) {
            auto dst = e_iter.get_neighbor();
            ctx.result_next[dst] = 1;
            total_edge_num++;
            ++e_iter;
          }
          
        ++iter;
      }
      if (frag.fid() == 0) {
        LOG(INFO) << "PEval stage prepare Query time: " << grape::GetCurrentTime() - start << " seconds";
      }
      std::cout << "total_edge_num = " << total_edge_num << " fid = " << frag.fid() << std::endl;
      return;
    }
    messages.InitChannels(1);
    
    int local_vertex_num = 0;
    local_vertex_num = frag.GetInnerVerticesNum();
    //std::cout << " local_vertex_num = " << local_vertex_num 
    //          << " fid = " << frag.fid() << std::endl;

    Sum(local_vertex_num, ctx.total_vertex_num);
    std::cout << "graph_vnum: " << ctx.total_vertex_num << std::endl;

    double p = 1.0 / ctx.total_vertex_num; 
    int dangling_vnum = 0;

    {
      double start = grape::GetCurrentTime();
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        ctx.degree[u] = frag.GetLocalOutDegree(u);
        ctx.result[u] = p;
        ++iter;
      }
      if (frag.fid() == 0) {
        LOG(INFO) << "PEval stage 0 Query time: " << grape::GetCurrentTime() - start << " seconds";
      }
    }
    {
      double start = grape::GetCurrentTime();
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        int edge_num = ctx.degree[u];
        if (edge_num > 0) {
          auto es = frag.GetOutgoingAdjList(u);
          auto e_iter = es.begin();
          while (!e_iter.is_end()) {
            auto dst = e_iter.get_neighbor();
            ctx.result_next[dst] += p / edge_num;
            ++e_iter;
          }
        } else {
          dangling_vnum++;
        }
        ++iter;
      }
      if (frag.fid() == 0) {
        LOG(INFO) << "PEval stage 1 Query time: " << grape::GetCurrentTime() - start << " seconds";
      }
    }
    Sum(dangling_vnum, ctx.total_dangling_vnum);
    ctx.dangling_sum = p * ctx.total_dangling_vnum;
    {
      double start = grape::GetCurrentTime();
      auto iter = outer_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        //std::cout << "vertex = " << " fid = " << frag.fid() << std::endl;
        messages.SyncStateOnOuterVertex(frag, u, ctx.result_next[u]);
        ctx.result_next[u] = 0.0;
        ++iter;
      }
      if (frag.fid() == 0) {
        LOG(INFO) << "PEval stage 2 Query time: " << grape::GetCurrentTime() - start << " seconds";
      }

    }
    //return;
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    LOG(INFO) << "IncEval";
    double start = grape::GetCurrentTime();
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    ctx.current_round++;

    double base = (1.0 - ctx.delta) / ctx.total_vertex_num +
                  ctx.delta * ctx.dangling_sum / ctx.total_vertex_num;
    ctx.dangling_sum = base * ctx.total_dangling_vnum;

    // process received ranks sent by other workers
    {
      messages.ParallelProcess<fragment_t, double>(
          thread_num(), frag, [&ctx](int tid, const vertex_t& u, const double& msg) {
            ctx.result_next[u] += msg;
          });
    }

    if (ctx.current_round == ctx.max_round) {
      auto iter = inner_vertices.begin();
      while (!iter.is_end()) {
        auto u = *iter;
        ctx.result[u] =
              base + ctx.delta * ctx.result_next[u];
        ++iter;
      }
    } else {
      {
        auto iter = inner_vertices.begin();
        while (!iter.is_end()) {
          auto u = *iter;
          ctx.result[u] =
                base + ctx.delta * ctx.result_next[u];
          ctx.result_next[u] = 0.0;
          ++iter;
        }
      } 
      {
        auto iter = inner_vertices.begin();
         while (!iter.is_end()) {
          auto src = *iter;
          int edge_num = ctx.degree[src];
          if (edge_num > 0) {
            double msg = ctx.result[src] / edge_num;
            auto es = frag.GetOutgoingAdjList(src);
            auto e_iter = es.begin();
            while (!e_iter.is_end()) {
              auto dst = e_iter.get_neighbor();
              ctx.result_next[dst] += msg;
              ++e_iter;
            }
          }
          ++iter;
         }
      }
      {
        auto iter = outer_vertices.begin();
        while (!iter.is_end()) {
          auto u = *iter;
          messages.SyncStateOnOuterVertex(frag, u, ctx.result_next[u]);
          ctx.result_next[u] = 0.0;
          ++iter;
        }
      }
      if (frag.fid() == 0) {
        LOG(INFO) << "IncEval Query time: " << grape::GetCurrentTime() - start << " seconds";
      }

    }

   
  }
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_APPS_PAGERANK_PAGERANK_NETWORKX_H_
