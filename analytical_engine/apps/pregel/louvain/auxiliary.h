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

template <typename VID_T, typename EDATA_T>
class LouvainNodeState {
  using vid_t = VID_T;
  using edata_t = EDATA_T;

 private:
  vid_t community_ = 0;
  edata_t community_sigma_total_;

  // the internal edge weight of a node
  // i.e. edges weight from the node to itself.
  edata_t internal_weight_;

  // degree of the node
  edata_t node_weight_;

  // 1 if the node has changed communities this cycle, otherwise 0
  int64_t changed_ = 0;

  // history of total change numbers, used to determine when to halt
  std::vector<int64_t> change_history_;

  bool from_louvain_vertex_reader_ = false;
  bool use_fake_edges_ = false;
  bool alived_community_ = true;
  std::map<vid_t, edata_t> fake_edges_;
  std::vector<vid_t> nodes_in_community_;

 public:
  LouvainNodeState()
      : community_(0),
        community_sigma_total_(0),
        internal_weight_(0),
        node_weight_(0),
        changed_(0) {}

  const vid_t get_community() const { return community_; }

  void set_community(const vid_t& community) { community_ = community; }

  edata_t get_community_sigma_total() const { return community_sigma_total_; }

  void set_community_sigma_total(edata_t community_sigma_total) {
    community_sigma_total_ = community_sigma_total;
  }
  edata_t get_internal_weight() const { return internal_weight_; }
  void set_internal_weight(edata_t internal_weight) {
    internal_weight_ = internal_weight;
  }
  edata_t get_node_weight() const { return node_weight_; }
  void set_node_weight(edata_t node_weight) { node_weight_ = node_weight; }
  int64_t get_changed() const { return changed_; }
  void set_changed(int64_t changed) { changed_ = changed; }
  const std::vector<int64_t>& get_change_history() const {
    return change_history_;
  }
  void set_change_history(const std::vector<int64_t>& change_history) {
    change_history_ = change_history;
  }
  bool is_from_louvain_vertex_reader() const {
    return from_louvain_vertex_reader_;
  }
  void set_from_louvain_vertex_reader(bool from_louvain_vertex_reader) {
    from_louvain_vertex_reader_ = from_louvain_vertex_reader;
  }

  void set_alived_community(bool alived) { alived_community_ = alived; }

  bool is_alived_community() { return alived_community_; }

  void add_change_history(int64_t change) { change_history_.push_back(change); }

  void clear_change_history() { change_history_.clear(); }

  bool use_fake_edges() { return use_fake_edges_; }

  const std::map<vid_t, edata_t>& get_fake_edges() { return fake_edges_; }

  void set_use_fake_edges(bool use_fake_edges) { use_fake_edges_ = use_fake_edges; }

  void set_fake_edges(const std::map<vid_t, edata_t>& edges) {
    fake_edges_ = edges;
  }

  std::vector<vid_t>& get_nodes_in_community() { return nodes_in_community_; }

  edata_t total_edge_weight = -1;
};

template <typename VID_T, typename EDATA_T>
class LouvainMessage {
  using vid_t = VID_T;
  using comm_id_type = VID_T;
  using edata_t = EDATA_T;

 public:
  comm_id_type community_id_;
  edata_t community_sigma_total_;
  edata_t edge_weight_;
  vid_t source_id_;
  vid_t dst_id;

  // For reconstruct graph info.
  // Each vertex send self's meta info to its community and silence it self,
  // the community compress its member's data and make self a new vertex for
  // next stage.
  edata_t internal_weight_ = 0;
  std::map<comm_id_type, edata_t> edges_;

 public:
  std::vector<comm_id_type> nodes_in_self_community;

 public:
  edata_t get_internal_weight() const { return internal_weight_; }
  void set_internal_weight(edata_t internal_weight) {
    internal_weight_ = internal_weight;
  }
  const std::map<comm_id_type, edata_t>& get_edges() const { return edges_; }
  void set_edges(const std::map<comm_id_type, edata_t>& edges) {
    edges_ = edges;
  }
  void set_edges(std::map<comm_id_type, edata_t>&& edges) { edges_ = edges; }

 public:
  LouvainMessage()
      : community_id_(0),
        community_sigma_total_(0),
        edge_weight_(0),
        source_id_(0),
        dst_id(0) {}
  LouvainMessage(const comm_id_type& community_id,
                 edata_t community_sigma_total, edata_t edge_weight,
                 const comm_id_type& source_id, const vid_t& dst_id)
      : community_id_(community_id),
        community_sigma_total_(community_sigma_total),
        edge_weight_(edge_weight),
        source_id_(source_id),
        dst_id(dst_id) {}

  void add_to_sigma_total(edata_t partial) {
    community_sigma_total_ += partial;
  }

  comm_id_type get_community_id() const { return community_id_; }
  void set_community_id(const comm_id_type& community_id) {
    community_id_ = community_id;
  }
  edata_t get_community_sigma_total() const { return community_sigma_total_; }
  void set_community_sigma_total(edata_t community_sigma_total) {
    community_sigma_total_ = community_sigma_total;
  }
  edata_t get_edge_weight() const { return edge_weight_; }
  void set_edge_weight(edata_t edge_weight) { edge_weight_ = edge_weight; }
  comm_id_type get_source_id() const { return source_id_; }
  void set_source_id(const comm_id_type& source_id) { source_id_ = source_id; }

  friend grape::InArchive& operator<<(grape::InArchive& in_archive,
                                      const LouvainMessage& u) {
    in_archive << u.community_id_;
    in_archive << u.community_sigma_total_;
    in_archive << u.edge_weight_;
    in_archive << u.source_id_;
    in_archive << u.dst_id;
    in_archive << u.internal_weight_;
    in_archive << u.edges_;
    in_archive << u.nodes_in_self_community;
    return in_archive;
  }
  friend grape::OutArchive& operator>>(grape::OutArchive& out_archive,
                                       LouvainMessage& val) {
    out_archive >> val.community_id_;
    out_archive >> val.community_sigma_total_;
    out_archive >> val.edge_weight_;
    out_archive >> val.source_id_;
    out_archive >> val.dst_id;
    out_archive >> val.internal_weight_;
    out_archive >> val.edges_;
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