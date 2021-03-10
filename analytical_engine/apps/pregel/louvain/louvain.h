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


#ifndef ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_H_
#define ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_H_

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "apps/pregel/louvain/auxiliary.h"
#include "apps/pregel/louvain/louvain_vertex.h"
#include "core/app/pregel/i_vertex_program.h"
#include "core/app/pregel/pregel_compute_context.h"

namespace gs {

template <typename FRAG_T>
class PregelLouvain
    : public IPregelProgram<LouvainVertex<FRAG_T, LouvainNodeState<typename FRAG_T::vid_t, typename FRAG_T::edata_t>, LouvainMessage<typename FRAG_T::vid_t, typename FRAG_T::edata_t>>,
                            PregelComputeContext<FRAG_T, LouvainNodeState<typename FRAG_T::vid_t, typename FRAG_T::edata_t>, LouvainMessage<typename FRAG_T::vid_t, typename FRAG_T::edata_t>>> {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vd_t = LouvainNodeState<vid_t, edata_t>;
  using md_t = LouvainMessage<vid_t, edata_t>;
  using pregel_vertex_t = LouvainVertex<fragment_t, vd_t, md_t>;
  using compute_context_t = PregelComputeContext<fragment_t, vd_t, md_t>;
  using comm_id_type = vid_t;

 public:
  void Init(pregel_vertex_t& v, compute_context_t& context) override {
    vd_t& vertex_value = v.ref_value();
    int64_t sigma_total = 0;
    for (auto& e : v.outgoing_edges()) {
        sigma_total += e.get_data();
    }
    for (auto& e : v.incoming_edges()) {
        sigma_total += e.get_data();
    }

    vertex_value.set_community(v.gid());
    vertex_value.set_community_sigma_total(sigma_total +
                                           vertex_value.get_internal_weight());
    vertex_value.set_node_weight(sigma_total);
    vertex_value.set_from_louvain_vertex_reader(true);
    v.nodes_in_self_community().push_back(vertex_value.get_community());

    // register aggregators
    context.register_aggregator(CHANGE_AGG,
                                PregelAggregatorType::kInt64SumAggregator);
    context.register_aggregator(TOTAL_EDGE_WEIGHT_AGG,
                                PregelAggregatorType::kInt64SumAggregator);
    context.register_aggregator(ACTUAL_Q_AGG,
                                PregelAggregatorType::kDoubleSumAggregator);
  }

  void Compute(grape::IteratorPair<md_t*> messages, pregel_vertex_t& v,
               compute_context_t& context) override {
    int current_super_step = context.superstep();
    // the step in this iteration
    int current_minor_step = current_super_step % 3;
    // the current iteration, two iterations make a full pass.
    int current_iteration = current_super_step / 3;

    vd_t& vertex_value = v.ref_value();

    if (current_super_step == -2) {
      send_communities_info(v);
      return;
    } else if (current_super_step == -1) {
      compress_communities(v, messages);
      return;
    }
    // count the total edge weight of the graph on the first super step only
    if (current_super_step == 0) {
      if (!vertex_value.is_from_louvain_vertex_reader()) {
        // not from the disk but from the previous round's result
        vertex_value.set_community(v.gid());
        int64_t edge_weight_aggregation = 0;
        // It must use fake edges since we already set them last round.
        for (auto& e : v.fake_edges()) {
          edge_weight_aggregation += e.second;
        }
        vertex_value.set_node_weight(edge_weight_aggregation);
      }
      vertex_value.total_edge_weight = -1;
      context.aggregate(TOTAL_EDGE_WEIGHT_AGG,
                        vertex_value.get_node_weight() + vertex_value.get_internal_weight());
    }

    if (current_super_step == 0 && v.edge_size() == 0) {
      // nodes that have no edges send themselves a message on the step 0
      md_t message;
      v.send_by_gid(v.gid(), message);
      v.vote_to_halt();
      return;
    } else if (current_super_step == 1 && v.edge_size() == 0) {
      // nodes that have no edges aggregate their Q value and exit computation on
      // step 1
      grape::IteratorPair<md_t*> msgs(NULL, NULL);
      double q = calculateActualQ(v, context, msgs);
      // LOG(INFO) << "nodes that have no edges exit with q: " << q;
      aggregateQ(context, q);
      v.vote_to_halt();
      return;
    }
    // at the start of each full pass check to see if progress is still being
    // made, if not halt
    if (current_minor_step == 1 && current_iteration > 0 &&
        current_iteration % 2 == 0) {
      vertex_value.set_changed(0);  // change count is per pass
      int64_t total_change = context.template get_aggregated_value<int64_t>(CHANGE_AGG);
      // LOG(INFO) << v.gid() << " get aggregate change " << total_change;
      vertex_value.add_change_history(total_change);

      // if halting aggregate q value and replace node edges with community edges
      // (for next stage in pipeline)
      if (decide_to_halt(vertex_value.get_change_history(),
                       std::stoi(context.get_config("tolerance")),
                       std::stoi(context.get_config("min_progress")))) {
        // stage 2
        double q = calculateActualQ(v, context, messages);
        replaceNodeEdgesWithCommunityEdges(v, messages);
        aggregateQ(context, q);
        vertex_value.clear_change_history();
        return;
        // note: we did not vote to halt, MasterCompute will halt computation on
        // next step
      }
    }

    switch (current_minor_step) {
    case 0:
      getAndSendCommunityInfo(v, context, messages);

      // in the next step will require a progress check, aggregate the number of
      // nodes who have changed community.
      if (current_iteration > 0 && current_iteration % 2 == 0) {
        // LOG(INFO) << v.gid() << " aggregate change " << vertex_value.get_changed();
        context.aggregate(CHANGE_AGG, vertex_value.get_changed());
      }
      break;
    case 1:
      calculateBestCommunity(v, context, messages, current_iteration);
      break;
    case 2:
      updateCommunities(v, messages);
      break;
    default:
      LOG(INFO) << "Invalid minor step: " << current_minor_step;
    }
    v.vote_to_halt();
  }

 private:
  void aggregateQ(compute_context_t& context, double q) {
    context.aggregate(ACTUAL_Q_AGG, static_cast<double>(q));
  }

  /**
  * Get the total edge weight of the graph.
  *
  * @return 2*the total graph weight.
  */
  int64_t getTotalEdgeWeight(compute_context_t& context, pregel_vertex_t& v) {
    auto& vertex_value = v.ref_value();
    if (vertex_value.total_edge_weight == -1) {
      int64_t total_edge_weight = context.template get_aggregated_value<int64_t>(TOTAL_EDGE_WEIGHT_AGG);
      vertex_value.total_edge_weight = total_edge_weight;
    }
    return vertex_value.total_edge_weight;
  }

  /**
  * Each vertex will receive its own communities sigma_total (if updated),
  * and then send its current community info to each of its neighbors.
  *
  * @param messages
  */
  void getAndSendCommunityInfo(pregel_vertex_t& vertex, compute_context_t& context,
                               const grape::IteratorPair<md_t*>& messages) {
    vd_t& state = vertex.ref_value();
    // set new community information.
    if (context.superstep() > 0) {
      if (messages.empty()) {
        LOG(ERROR) << "Error! No community info received in "
                      "getAndSendCommunityInfo! Super step: "
                   << context.superstep() << " id: " << vertex.id();
      }
      if (messages.size() > 1) {
        LOG(ERROR) << "Error! More than one community info packets received in "
                    "getAndSendCommunityInfo! Super step: "
                   << context.superstep() << " id: " << vertex.id();
      }
      state.set_community(messages.begin()->get_community_id());
      state.set_community_sigma_total(
            messages.begin()->get_community_sigma_total());
    }
    if (vertex.use_fake_edges()) {
      for (const auto& edge : vertex.fake_edges()) {
        md_t out_message(state.get_community(),
                         state.get_community_sigma_total(), edge.second,
                         vertex.gid());
        vertex.send_by_gid(edge.first, out_message);
      }
    } else {
      for (auto& edge : vertex.incoming_edges()) {
        md_t out_message(state.get_community(),
                         state.get_community_sigma_total(), edge.get_data(),
                         vertex.gid());
        vertex.send(edge.get_neighbor(), out_message);
      }
      for (auto& edge : vertex.outgoing_edges()) {
        md_t out_message(state.get_community(),
                         state.get_community_sigma_total(), edge.get_data(),
                         vertex.gid());
        vertex.send(edge.get_neighbor(), out_message);
      }
    }
  }

  /**
  * Based on community of each of its neighbors, each vertex determines if
  * it should retain its current community or switch to a neighboring
  * community.
  * <p/>
  * At the end of this step a message is sent to the nodes community hub so a
  * new community sigma_total can be calculated.
  *
  * @param messages
  * @param iteration
  */
  void calculateBestCommunity(pregel_vertex_t& vertex, compute_context_t& context,
                              const grape::IteratorPair<md_t*>& messages,
                              int iteration) {
    // group messages by communities.
    std::map<comm_id_type, md_t> community_map;
    for (auto& message : messages) {
      comm_id_type community_id = message.get_community_id();
      int64_t weight = message.get_edge_weight();
      if (community_map.find(community_id) != community_map.end()) {
        community_map[community_id].set_edge_weight(
            community_map[community_id].get_edge_weight() + weight);
      } else {
        community_map[community_id] = message;
      }
    }

    auto& state = vertex.ref_value();
    // calculate change in Q for each potential community
    comm_id_type best_community_id = state.get_community();
    comm_id_type starting_community_id = best_community_id;
    double max_delta_Q = 0.0;
    for (auto& entry : community_map) {
      double deltaQ = calculateQDelta(
          context, vertex, starting_community_id, entry.second.get_community_id(),
          entry.second.get_community_sigma_total(),
          entry.second.get_edge_weight(), state.get_node_weight(),
          state.get_internal_weight());
      // TODO: double compare, string compare
      if (deltaQ > max_delta_Q ||
          (deltaQ == max_delta_Q &&
          entry.second.get_community_id() < best_community_id)) {
        best_community_id = entry.second.get_community_id();
        max_delta_Q = deltaQ;
      }
    }

    // ignore switches based on iteration (prevent certain cycles)
    if ((state.get_community() > best_community_id && iteration % 2 == 0) ||
        (state.get_community() < best_community_id && iteration % 2 != 0)) {
      best_community_id = state.get_community();
    }

    // update community and change count
    if (state.get_community() != best_community_id) {
      // comm_id_type old = state.get_community();
      md_t c = community_map[best_community_id];
      if (best_community_id != c.get_community_id()) {
        LOG(ERROR) << "Error! Community mapping contains wrong Id";
      }
      state.set_community(c.get_community_id());
      state.set_community_sigma_total(c.get_community_sigma_total());
      state.set_changed(1);
    }
    // send our node weight to the community hub to be summed in next super step
    md_t message(state.get_community(),
                 state.get_node_weight() + state.get_internal_weight(), 0,
                 vertex.gid());
    // LOG(INFO) << "Sending node weight to the community hub";
    vertex.send_by_gid(state.get_community(), message);
  }

  /**
  * determine the change in q if a node were to move to the given community.
  *
  * @param currCommunityId
  * @param testCommunityId
  * @param testSigmaTotal
  * @param edgeWeightInCommunity (sum of weight of edges from this node to target
  * community)
  * @param nodeWeight            (the node degree)
  * @param internalWeight
  * @return
  */
  // TODO: big decimal
  double calculateQDelta(compute_context_t& context, pregel_vertex_t& v,
                         const comm_id_type& curr_community_id,
                         const comm_id_type& test_community_id,
                         int64_t test_sigma_total,
                         int64_t edge_weight_in_community, int64_t node_weight,
                         int64_t internal_weight) {
    bool is_current_community = (curr_community_id == test_community_id);
    double M = getTotalEdgeWeight(context, v);
    int64_t k_i_in_L = is_current_community
                         ? edge_weight_in_community + internal_weight
                         : edge_weight_in_community;
    double k_i_in = k_i_in_L;
    double k_i = node_weight + internal_weight;
    double sigma_tot = test_sigma_total;
    if (is_current_community) {
      sigma_tot -= k_i;
    }

    double deltaQ = 0.0;
    if (!(is_current_community && sigma_tot == deltaQ)) {
      double dividend = k_i * sigma_tot;
      deltaQ = k_i_in - dividend / M;
    }
    return deltaQ;
  }

  /**
  * Each community hub aggregates the values from each of its members to
  * update the node's sigma total, and then sends this back to each of its
  * members.
  *
  * @param messages
  */
  void updateCommunities(pregel_vertex_t& vertex,
                         const grape::IteratorPair<md_t*>& messages) {
    // sum all community contributions
    md_t sum;
    sum.set_community_id(vertex.gid());
    sum.set_community_sigma_total(0);
    for (auto& m : messages) {
      sum.add_to_sigma_total(m.get_community_sigma_total());
    }

    for (auto& m : messages) {
      vertex.send_by_gid(m.get_source_id(), sum);
    }
  }

  /**
   * Calculate this nodes contribution for the actual q value of the graph.
   */
  double calculateActualQ(pregel_vertex_t& vertex, compute_context_t& context,
                          const grape::IteratorPair<md_t*>& messages) {
    auto state = vertex.value();
    int64_t k_i_in = state.get_internal_weight();
    for (auto& m : messages) {
      if (m.get_community_id() == state.get_community()) {
        k_i_in += vertex.get_edge_value(m.get_source_id());
      }
    }
    int64_t sigma_tot = state.get_community_sigma_total();
    int64_t M = getTotalEdgeWeight(context, vertex);
    int64_t k_i = state.get_node_weight() + state.get_internal_weight();

    double q = static_cast<double>(k_i_in) / M -
               static_cast<double>(sigma_tot * k_i) / pow(M, 2);
    q = q < 0 ? 0 : q;
    return q;
  }

  /**
   * Replace each edge to a neighbor with an edge to that neighbors community
   * instead. Done just before exiting computation. In the next state of the
   * pipe line this edges are aggregated and all communities are represented
   * as single nodes. Edges from the community to itself are tracked be the
   * nodes internal weight.
   *
   * @param messages
   */
  void replaceNodeEdgesWithCommunityEdges(
      pregel_vertex_t& vertex, grape::IteratorPair<md_t*>& messages) {
    std::map<comm_id_type, edata_t> community_map;
    for (auto& message : messages) {
      const auto& community_id = message.get_community_id();
      community_map[community_id] += message.get_edge_weight();
    }

    vertex.set_fake_edges(std::move(community_map));
  }

  void send_communities_info(pregel_vertex_t& vertex) {
    vd_t& vertex_value = vertex.ref_value();
    md_t message;
    message.set_internal_weight(vertex_value.get_internal_weight());
    std::map<comm_id_type, edata_t> edges;
    if (vertex.use_fake_edges()) {
      edges = vertex.fake_edges();
    } else {
      LOG(ERROR) << "Not supposed to be here.";
      for (auto& edge : vertex.incoming_edges()) {
        edges[vertex.get_vertex_gid(edge.get_neighbor())] = edge.get_data();
      }
      for (auto& edge : vertex.outgoing_edges()) {
        edges[vertex.get_vertex_gid(edge.get_neighbor())] = edge.get_data();
      }
    }
    message.set_edges(std::move(edges));
    if (vertex.gid() != vertex_value.get_community()) {
      message.nodes_in_self_community.swap(vertex.nodes_in_self_community());
    }
    vertex.send_by_gid(vertex_value.get_community(), message);

    vertex.vote_to_halt();
  }

  void compress_communities(pregel_vertex_t& vertex,
                            grape::IteratorPair<md_t*>& messages) {
    auto community_id = vertex.gid();
    int64_t weight = 0;
    std::map<comm_id_type, edata_t> edge_map;
    auto& nodes_in_self_community = vertex.nodes_in_self_community();
    for (auto& m : messages) {
      weight += m.get_internal_weight();
      for (auto& entry : m.get_edges()) {
        if (entry.first == community_id) {
          weight += entry.second;
        } else {
          edge_map[entry.first] += entry.second;
        }
      }
      nodes_in_self_community.insert(nodes_in_self_community.end(),
                                     m.nodes_in_self_community.begin(),
                                     m.nodes_in_self_community.end());
    }
    vertex.ref_value().set_internal_weight(weight);
    vertex.set_fake_edges(std::move(edge_map));
    vertex.ref_value().set_from_louvain_vertex_reader(false);

    // send self fake message to activate next round.
    md_t fake_message;
    vertex.send_by_gid(community_id, fake_message);
    // do not vote to halt since next round those new vertex need to be active.
  }

};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_H_
