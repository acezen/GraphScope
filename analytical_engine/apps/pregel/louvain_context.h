#ifndef ANALYTICAL_ENGINE_CORE_APP_PREGEL_LOUVAIN_CONTEXT_H_
#define ANALYTICAL_ENGINE_CORE_APP_PREGEL_LOUVAIN_CONTEXT_H_

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
  using vid_t = typename FRAG_T::vid_t;
  using edata_t = typename FRAG_T::edata_t;
  using vd_t = typename COMPUTE_CONTEXT_T::vd_t;
  using fragment_t = FRAG_T;

 public:
  explicit LouvainContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, typename COMPUTE_CONTEXT_T::vd_t>(
            fragment),
        compute_context_(this->data()) {}

  void Init(grape::DefaultMessageManager& messages, const std::string& args) {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    compute_context_.init(frag);
    compute_context_.set_fragment(&frag);
    compute_context_.set_message_manager(&messages);

    if (!args.empty()) {
      // The app params are passed via serialized json string.
      boost::property_tree::ptree pt;
      std::stringstream ss;
      ss << args;
      boost::property_tree::read_json(ss, pt);
      for (const auto& x : pt) {
        compute_context_.set_config(x.first, x.second.get_value<std::string>());
      }
    }
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto& result = compute_context_.vertex_data();
    auto iv = frag.InnerVertices();
    for (auto v : iv) {
      auto& list = result[v].get_nodes_in_community();
      if (!list.empty()) {
        auto community_id = frag.Gid2Oid(list.front());
        for (auto& gid : list) {
          os << frag.Gid2Oid(gid) << " " << community_id << std::endl;
        }
      }
    }
  }

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

  std::vector<int64_t> change_history;
  bool halt = false;  // stage 1
  COMPUTE_CONTEXT_T compute_context_;
  double previous_q = 0.0;
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_CORE_APP_PREGEL_LOUVAIN_CONTEXT_H_
