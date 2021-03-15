#ifndef ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_CONTEXT_H_
#define ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_CONTEXT_H_

#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"

#include "grape/grape.h"

#include "core/context/vertex_data_context.h"

namespace gs {

template <typename FRAG_T, typename COMPUTE_CONTEXT_T>
class LouvainContext
    : public grape::VertexDataContext<FRAG_T,
                                      typename COMPUTE_CONTEXT_T::vd_t> {
  using fragment_t = FRAG_T;
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using edata_t = typename FRAG_T::edata_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using state_t = LouvainNodeState<vid_t, edata_t>;
  using vertex_state_array_t = typename fragment_t::template vertex_array_t<state_t>;

 public:
  explicit LouvainContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, typename COMPUTE_CONTEXT_T::vd_t>(
            fragment),
        compute_context_(this->data()) {}

  void Init(grape::DefaultMessageManager& messages, int tol, int min_p) {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    compute_context_.init(frag);
    compute_context_.set_fragment(&frag);
    compute_context_.set_message_manager(&messages);

    tolerance = tol;
    min_progress = min_p;
    vertex_state_.Init(inner_vertices);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    // auto& result = compute_context_.vertex_data();
    auto iv = frag.InnerVertices();
    for (auto v : iv) {
      auto& list = vertex_state_[v].nodes_in_community;
      // os << frag.GetId(v) << " " << result[v] << std::endl;
      if (!list.empty()) {
        auto community_id = frag.Gid2Oid(list.front());
        for (auto& gid : list) {
          os << frag.Gid2Oid(gid) << " " << community_id << std::endl;
        }
      }
    }
  }

  void SyncCommunity(grape::DefaultMessageManager& messages) {
    auto& frag = this->fragment();
    auto& vid_parser = compute_context_.get_vid_parser();
    auto& comm_result = compute_context_.vertex_data();
    auto iv = frag.InnerVertices();
    for (auto v : iv) {
      auto& list = vertex_state_[v].nodes_in_community;
      if (!list.empty()) {
        auto community_id = frag.Gid2Oid(list.front());
        for (auto& gid : list) {
          auto fid = vid_parser.GetFid(gid);
          vertex_t lv;
          if (fid == frag.fid()) {
            frag.InnerVertexGid2Vertex(gid, lv);
            comm_result[lv] = community_id;
          } else {
            messages.SendToFragment(
              fid,
              std::pair<vid_t, oid_t>(gid, community_id)
            );
          }
        }
      }
    }
  }

  state_t& GetVertexState(const vertex_t& v) {
    return vertex_state_[v];
  }

  std::vector<int64_t> change_history;
  bool halt = false;  // stage 1
  COMPUTE_CONTEXT_T compute_context_;
  double previous_q = 0.0;
  int tolerance;
  int min_progress;

  vertex_state_array_t vertex_state_;
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_PREGEL_LOUVAIN_LOUVAIN_CONTEXT_H_
