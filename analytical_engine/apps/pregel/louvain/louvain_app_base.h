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

#ifndef ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_APP_BASE_H_
#define ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_APP_BASE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "grape/grape.h"
#include "grape/utils/iterator_pair.h"

#include "core/app/app_base.h"
#include "core/app/pregel/pregel_compute_context.h"

#include "apps/pregel/louvain/auxiliary.h"
#include "apps/pregel/louvain/louvain_context.h"
#include "apps/pregel/louvain/louvain.h"
#include "apps/pregel/louvain/louvain_vertex.h"

namespace gs {

/**
 * @brief This class is a specialized PregelAppBase for louvain.
 * @tparam FRAG_T
 */
template <typename FRAG_T, typename VERTEX_PROGRAM_T = PregelLouvain<FRAG_T>>
class LouvainAppBase
    : public AppBase<
          FRAG_T,
          LouvainContext<FRAG_T, PregelComputeContext<
                                    FRAG_T, typename VERTEX_PROGRAM_T::vd_t,
                                    typename VERTEX_PROGRAM_T::md_t>>>,
      public grape::Communicator {
  using app_t = LouvainAppBase<FRAG_T>;
  using vertex_program_t = VERTEX_PROGRAM_T;
  using vd_t = typename vertex_program_t::vd_t;
  using md_t = typename vertex_program_t::md_t;
  using pregel_compute_context_t = PregelComputeContext<FRAG_T, vd_t, md_t>;
  using pregel_context_t = LouvainContext<FRAG_T, pregel_compute_context_t>;

  INSTALL_DEFAULT_WORKER(app_t, pregel_context_t, FRAG_T)

 public:
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;

  explicit LouvainAppBase(const vertex_program_t& program = vertex_program_t())
        : program_(program) {}

  void PEval(const fragment_t& frag, pregel_context_t& ctx,
             message_manager_t& messages) {
    // superstep is 0 in PEval
    LouvainVertex<fragment_t, vd_t, md_t> pregel_vertex;
    pregel_vertex.set_context(&ctx);
    pregel_vertex.set_fragment(&frag);
    pregel_vertex.set_compute_context(&ctx.compute_context_);
    context.register_aggregator(ALL_HALTED,
                                PregelAggregatorType::kBoolAndAggregator);

    grape::IteratorPair<md_t*> null_messages(nullptr, nullptr);
    auto inner_vertices = frag.InnerVertices();

    for (auto v : inner_vertices) {
      pregel_vertex.set_vertex(v);
      program_.Init(pregel_vertex, ctx.compute_context_);
    }

    for (auto v : inner_vertices) {
      pregel_vertex.set_vertex(v);
      program_.Compute(null_messages, pregel_vertex, ctx.compute_context_);
    }

    {
      // Sync Aggregator
      for (auto& pair : ctx.compute_context_.aggregators()) {
        grape::InArchive iarc;
        std::vector<grape::InArchive> oarcs;
        std::string name = pair.first;
        pair.second->Serialize(iarc);
        pair.second->Reset();
        AllGather(std::move(iarc), oarcs);
        pair.second->DeserializeAndAggregate(oarcs);
        pair.second->StartNewRound();
      }
    }

    ctx.compute_context_.clear_for_next_round();

    if (!ctx.compute_context_.all_halted()) {
      messages.ForceContinue();
    }
  }

  void IncEval(const fragment_t& frag, pregel_context_t& ctx,
               message_manager_t& messages) {
    ctx.compute_context_.inc_step();

    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    int current_super_step = ctx.compute_context_.superstep();
    int current_minor_step = current_super_step % 3;
    int current_iteration = current_super_step / 3;

    LOG(INFO) << "current super step: " << current_super_step
              << " current minor step: " << current_minor_step
              << " current iteration: " << current_iteration;

    if (ctx.template get_aggregated_value<bool>(ALL_HALTED)) {
      LOG(INFO) << "all workers halted.";
      ctx.compute_context_.set_superstep(-10);
      ctx.SyncCommunity(messages);
      return;
    }
    {
      if (current_super_step == -9) {
        std::pair<vid_t, oid_t> msg;
        while (messages.GetMessage<std::pair<vid_t, oid_t>>(msg)) {
          vertex_t v;
          vid_t v_vid = msg.first;
          oid_t comm_id = msg.second;
          frag.InnerVertexGid2Vertex(v_vid, v);
          ctx.compute_context_.vertex_data()[v] = comm_id;
        }
        return;
      } else {
        // get message
        md_t msg;
        while (messages.GetMessage<md_t>(msg)) {
          vertex_t v;
          if(!frag.InnerVertexGid2Vertex(msg.dst_id, v)) {
            LOG(FATAL) << "v-" << msg.dst_id << " is not inner vertex of frag-"
                      << frag.fid();
          }
          ctx.compute_context_.messages_in()[v].emplace_back(std::move(msg));
          ctx.compute_context_.activate(v);
        }
      }
    }

    LouvainVertex<fragment_t, vd_t, md_t> pregel_vertex;
    pregel_vertex.set_fragment(&frag);
    pregel_vertex.set_compute_context(&ctx.compute_context_);
    pregel_vertex.set_context(&ctx);

    if (current_minor_step == 1 && current_iteration > 0 &&
        current_iteration % 2 == 0) {
      int64_t totalChange =
          ctx.compute_context_.template get_aggregated_value<int64_t>(
              CHANGE_AGG);
      ctx.change_history.push_back(totalChange);
      ctx.halt = decide_to_halt(
          ctx.change_history,
          ctx.tolerance,
          ctx.min_progress);
      if (ctx.halt) {
        VLOG(2) << "super step " << current_super_step << " decided to halt.";
      }
      VLOG(2) << "[INFO]: superstep: " << current_super_step
                << " pass: " << current_iteration / 2
                << " totalChange: " << totalChange;
    } else if (ctx.halt && current_super_step > -9) {
      double actualQ =
          ctx.compute_context_.template get_aggregated_value<double>(
              ACTUAL_Q_AGG);
      // after one pass if already decided halt, that means stage 1 yield no
      // changes, so we halt stage 2.
      if (current_super_step <= 14 || actualQ <= ctx.previous_q) {
        // stage 2 halt, the whole computaion terminated.
        if (actualQ < ctx.previous_q) {
          LOG(INFO) << "stage 2 halt, Final Q: " << ctx.previous_q;
        } else {
          LOG(INFO) << "stage 2 halt, Final Q: " << actualQ;
        }
      } else if (ctx.compute_context_.superstep() > 0) {
        // stage 1 halt
        VLOG(2) << "super step: " << current_super_step
                  << " decided to halt, ACTUAL Q: " << actualQ
                  << " previous Q: " << ctx.previous_q;
        ctx.compute_context_.set_superstep(-2);

        ctx.previous_q = actualQ;
        ctx.change_history.clear();
        ctx.halt = false;
      }
    }
    // At the start of each round, every alive node send to their
    // communities node info.
    if (ctx.compute_context_.superstep() == -2) {
      for (auto& v : inner_vertices) {
        if (ctx.GetVertexState(v).is_alived_community) {
          ctx.compute_context_.activate(v);
        }
      }
    }


    if (current_super_step > -9) {
    for (auto v : inner_vertices) {
      if (ctx.compute_context_.active(v)) {
        pregel_vertex.set_vertex(v);
        auto& cur_msgs = (ctx.compute_context_.messages_in())[v];
        program_.Compute(
            grape::IteratorPair<md_t*>(
                &cur_msgs[0],
                &cur_msgs[0] + static_cast<ptrdiff_t>(cur_msgs.size())),
            pregel_vertex, ctx.compute_context_);
      } else if (ctx.compute_context_.superstep() == -1) {
        ctx.GetVertexState(v).is_alived_community = false;
      }
    }
    }

    {
      // Sync Aggregator
      for (auto& pair : ctx.compute_context_.aggregators()) {
        grape::InArchive iarc;
        std::vector<grape::InArchive> oarcs;
        std::string name = pair.first;
        pair.second->Serialize(iarc);
        pair.second->Reset();
        AllGather(std::move(iarc), oarcs);
        pair.second->DeserializeAndAggregate(oarcs);
        pair.second->StartNewRound();
      }
    }

    ctx.compute_context_.clear_for_next_round();
    if (!ctx.compute_context_.all_halted()) {
      ctx.aggregate(ALL_HALTED, false);
      messages.ForceContinue();
    } else if (current_super_step != -9) {
      ctx.aggregate(ALL_HALTED, true);
      LOG(INFO) << "frag-" << frag.fid() << " halted.";
    }
  }

 private:
  vertex_program_t program_;
};
}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_APP_BASE_H_
