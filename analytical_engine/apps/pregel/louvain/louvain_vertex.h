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

#ifndef ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_VERTEX_H_
#define ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_VERTEX_H_

#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/app/pregel/pregel_vertex.h"

namespace gs {

template <typename FRAG_T, typename VD_T, typename MD_T>
class LouvainVertex : public PregelVertex<FRAG_T, VD_T, MD_T> {
  using fragment_t = FRAG_T;
  using vertex_t = typename fragment_t::vertex_t;
  using adj_list_t = typename fragment_t::const_adj_list_t;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using compute_context_t = PregelComputeContext<fragment_t, VD_T, MD_T>;
  using context_t = LouvainContext<FRAG_T, compute_context_t>;
  using state_t = LouvainNodeState<vid_t, edata_t>;

 public:
  using vd_t = VD_T;
  using md_t = MD_T;

  std::string id() { return std::to_string(fragment_->GetId(vertex_)); }

  void set_value(const VD_T& value) {
    compute_context_->set_vertex_value(*this, value);
  }
  void set_value(const VD_T&& value) {
    compute_context_->set_vertex_value(*this, std::move(value));
  }

  const VD_T& value() { return compute_context_->get_vertex_value(*this); }

  vertex_t vertex() const { return vertex_; }

  adj_list_t outgoing_edges() { return fragment_->GetOutgoingAdjList(vertex_); }

  adj_list_t incoming_edges() { return fragment_->GetIncomingAdjList(vertex_); }

  void send(const vertex_t& v, const MD_T& value) {
    compute_context_->send_message(v, value);
  }

  void send(const vertex_t& v, MD_T&& value) {
    compute_context_->send_message(v, std::move(value));
  }

  void vote_to_halt() { compute_context_->vote_to_halt(*this); }

  void set_fragment(const fragment_t* fragment) { fragment_ = fragment; }

  void set_compute_context(compute_context_t* compute_context) {
    compute_context_ = compute_context;
  }

  void set_context(context_t* context) {
    context_ = context;
  }

  void set_vertex(vertex_t vertex) { vertex_ = vertex; }

  state_t& ref_state() {
    return context_->GetVertexState(vertex_);
  }

  vid_t gid() {
    return fragment_->Vertex2Gid(vertex_);
  }

  vid_t get_vertex_gid(const vertex_t& v) { return fragment_->Vertex2Gid(v); }

  void send_by_id(const oid_t& dst_id, const md_t& md) {
    vertex_t v;
    fragment_->GetVertex(dst_id, v);
    send(v, md);
  }

  void send_by_gid(vid_t dst_gid, const md_t& md) {
    compute_context_->send_p2p_message(dst_gid, md);
  }

  size_t edge_size() {
    if (!use_fake_edges()) {
      return this->incoming_edges().Size() + this->outgoing_edges().Size();
    } else {
      return fake_edges().size();
    }
  }

  bool use_fake_edges() {
    return context_->GetVertexState(vertex_).use_fake_edges();
  }

  const std::map<vid_t, edata_t>& fake_edges() const {
    return context_->GetVertexState(vertex_).get_fake_edges();
  }

  // TODO: const reference
  edata_t get_edge_value(const vid_t& dst_id) {
    if (!use_fake_edges()) {
      for (auto& edge : this->incoming_edges()) {
        if (fragment_->Vertex2Gid(edge.get_neighbor()) == dst_id) {
          return edge.get_data();
        }
      }
      for (auto& edge : this->outgoing_edges()) {
        if (fragment_->Vertex2Gid(edge.get_neighbor()) == dst_id) {
          return edge.get_data();
        }
      }
    } else {
      return fake_edges().at(dst_id);
    }
    return edata_t();
  }

  void set_fake_edges(std::map<vid_t, edata_t>&& edges) {
    state_t& ref_state = this->ref_state();
    ref_state.set_fake_edges(edges);
    ref_state.set_use_fake_edges(true);
  }

  std::vector<vid_t>& nodes_in_self_community() {
    return context_->GetVertexState(vertex_).get_nodes_in_community();
  }

 public:
  const fragment_t* fragment_;
  PregelComputeContext<fragment_t, VD_T, MD_T>* compute_context_;
  context_t* context_;

  vertex_t vertex_;

};
}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_VERTEX_H_
