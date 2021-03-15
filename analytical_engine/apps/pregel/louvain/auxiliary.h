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


#ifndef ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_AUXILIARY_H_
#define ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_AUXILIARY_H_

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "grape/grape.h"

namespace gs {

#define CHANGE_AGG "change_aggregator"
#define TOTAL_EDGE_WEIGHT_AGG "total_edge_weight_aggregator"
#define ACTUAL_Q_AGG "actual_q_aggregator"
#define ALL_HALTED "all_halted"

template <typename VID_T, typename EDATA_T>
struct LouvainNodeState {
  using vid_t = VID_T;
  using edata_t = EDATA_T;

  vid_t community = 0;
  edata_t community_sigma_total;

  // the internal edge weight of a node
  // i.e. edges weight from the node to itself.
  edata_t internal_weight;

  // degree of the node
  edata_t node_weight;

  // 1 if the node has changed communities this cycle, otherwise 0
  int64_t changed;
  bool reset_total_edge_weight;

  bool is_from_louvain_vertex_reader = false;
  bool use_fake_edges = false;
  bool is_alived_community = true;
  std::map<vid_t, edata_t> fake_edges;
  std::vector<vid_t> nodes_in_community;
  edata_t total_edge_weight;

 public:
  LouvainNodeState()
      : community(0),
        community_sigma_total(0),
        internal_weight(0),
        node_weight(0),
        changed(0),
        reset_total_edge_weight(false),
        is_from_louvain_vertex_reader(false),
        use_fake_edges(false),
        is_alived_community(true) {}
};

template <typename VID_T, typename EDATA_T>
struct LouvainMessage {
  using vid_t = VID_T;
  using edata_t = EDATA_T;

  vid_t community_id;
  edata_t community_sigma_total;
  edata_t edge_weight;
  vid_t source_id;
  vid_t dst_id;

  // For reconstruct graph info.
  // Each vertex send self's meta info to its community and silence it self,
  // the community compress its member's data and make self a new vertex for
  // next stage.
  edata_t internal_weight = 0;
  std::map<vid_t, edata_t> edges;
  std::vector<vid_t> nodes_in_self_community;

 public:
  LouvainMessage()
      : community_id(0),
        community_sigma_total(0),
        edge_weight(0),
        source_id(0),
        dst_id(0) {}

  LouvainMessage(const vid_t& community_id,
                 edata_t community_sigma_total, edata_t edge_weight,
                 const vid_t& source_id, const vid_t& dst_id)
      : community_id(community_id),
        community_sigma_total(community_sigma_total),
        edge_weight(edge_weight),
        source_id(source_id),
        dst_id(dst_id) {}

  friend grape::InArchive& operator<<(grape::InArchive& in_archive,
                                      const LouvainMessage& u) {
    in_archive << u.community_id;
    in_archive << u.community_sigma_total;
    in_archive << u.edge_weight;
    in_archive << u.source_id;
    in_archive << u.dst_id;
    in_archive << u.internal_weight;
    in_archive << u.edges;
    in_archive << u.nodes_in_self_community;
    return in_archive;
  }
  friend grape::OutArchive& operator>>(grape::OutArchive& out_archive,
                                       LouvainMessage& val) {
    out_archive >> val.community_id;
    out_archive >> val.community_sigma_total;
    out_archive >> val.edge_weight;
    out_archive >> val.source_id;
    out_archive >> val.dst_id;
    out_archive >> val.internal_weight;
    out_archive >> val.edges;
    out_archive >> val.nodes_in_self_community;
    return out_archive;
  }
};

bool decide_to_halt(const std::vector<int64_t>& history, int tolerance,
                    int min_progress) {
  // Halt if the most recent change was 0
  if (0 == history.back()) {
    return true;
  }
  // Halt if the change count has increased tolerant times
  int64_t previous = history.front();
  int count = 0;
  for (auto& cur : history) {
    if (cur >= previous - min_progress) {
      count++;
    }
    previous = cur;
  }
  return (count > tolerance);
}

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_AUXILIARY_H_