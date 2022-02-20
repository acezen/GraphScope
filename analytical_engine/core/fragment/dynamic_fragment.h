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

#ifndef ANALYTICAL_ENGINE_CORE_FRAGMENT_DYNAMIC_FRAGMENT_H_
#define ANALYTICAL_ENGINE_CORE_FRAGMENT_DYNAMIC_FRAGMENT_H_

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "grape/communication/communicator.h"
#include "grape/fragment/basic_fragment_mutator.h"
#include "grape/fragment/csr_edgecut_fragment_base.h"
#include "grape/util.h"
#include "grape/utils/bitset.h"
#include "grape/utils/vertex_set.h"

#include "core/fragment/de_mutable_csr.h"
#include "core/object/dynamic.h"
#include "core/utils/partitioner.h"
#include "proto/graphscope/proto/types.pb.h"

namespace gs {

struct DynamicFragmentTraits {
  using oid_t = dynamic::Value;
  using vid_t = vineyard::property_graph_types::VID_TYPE;
  using vdata_t = dynamic::Value;
  using edata_t = dynamic::Value;
  using nbr_t = grape::Nbr<vid_t, edata_t>;
  using vertex_map_t = grape::GlobalVertexMap<oid_t, vid_t>;
  using inner_vertices_t = grape::VertexRange<vid_t>;
  using outer_vertices_t = grape::VertexRange<vid_t>;
  using vertices_t = grape::DualVertexRange<vid_t>;
  using sub_vertices_t = grape::VertexVector<vid_t>;

  using fragment_adj_list_t =
      grape::FilterAdjList<vid_t, edata_t, std::function<bool(const nbr_t&)>>;
  using fragment_const_adj_list_t =
      grape::FilterConstAdjList<vid_t, edata_t,
                                std::function<bool(const nbr_t&)>>;

  using csr_t = grape::DeMutableCSR<vid_t, nbr_t>;
  using csr_builder_t = grape::DeMutableCSRBuilder<vid_t, nbr_t>;
  using mirror_vertices_t = std::vector<grape::Vertex<vid_t>>;
};

class DynamicFragment
    : public grape::CSREdgecutFragmentBase<
          dynamic::Value, vineyard::property_graph_types::VID_TYPE,
          dynamic::Value, dynamic::Value, DynamicFragmentTraits> {
 public:
  using oid_t = dynamic::Value;
  using vid_t = vineyard::property_graph_types::VID_TYPE;
  using vdata_t = dynamic::Value;
  using edata_t = dynamic::Value;
  using traits_t = DynamicFragmentTraits;
  using base_t =
      grape::CSREdgecutFragmentBase<oid_t, vid_t, vdata_t, edata_t, traits_t>;
  using internal_vertex_t = grape::internal::Vertex<vid_t, vdata_t>;
  using edge_t = grape::Edge<vid_t, edata_t>;
  using nbr_t = grape::Nbr<vid_t, edata_t>;
  using vertex_t = grape::Vertex<vid_t>;

  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  using vertex_map_t = typename traits_t::vertex_map_t;
  using partitioner_t = typename vertex_map_t::partitioner_t;
  using mutation_t = grape::Mutation<vid_t, vdata_t, edata_t>;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

  using inner_vertices_t = typename traits_t::inner_vertices_t;
  using outer_vertices_t = typename traits_t::outer_vertices_t;
  using vertices_t = typename traits_t::vertices_t;
  using fragment_adj_list_t = typename traits_t::fragment_adj_list_t;
  using fragment_const_adj_list_t =
      typename traits_t::fragment_const_adj_list_t;

  template <typename T>
  using inner_vertex_array_t = grape::VertexArray<inner_vertices_t, T>;

  template <typename T>
  using outer_vertex_array_t = grape::VertexArray<outer_vertices_t, T>;

  template <typename T>
  using vertex_array_t = grape::VertexArray<vertices_t, T>;

  using vertex_range_t = inner_vertices_t;

  explicit DynamicFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : grape::FragmentBase<oid_t, vid_t, vdata_t, edata_t, traits_t>(vm_ptr) {}
  virtual ~DynamicFragment() = default;

  using base_t::buildCSR;
  using base_t::init;
  using base_t::IsInnerVertexGid;
  void Init(fid_t fid, bool directed, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) override {
    init(fid, directed);

    load_strategy_ = directed ? grape::LoadStrategy::kBothOutIn
                              : grape::LoadStrategy::kOnlyIn;

    ovnum_ = 0;
    static constexpr vid_t invalid_vid = std::numeric_limits<vid_t>::max();
    if (load_strategy_ == grape::LoadStrategy::kOnlyIn) {
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.dst)) {
          if (!IsInnerVertexGid(e.src)) {
            parseOrAddOuterVertexGid(e.src);
          }
        } else {
          e.src = invalid_vid;
        }
      }
    } else if (load_strategy_ == grape::LoadStrategy::kOnlyOut) {
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.dst);
          }
        } else {
          e.src = invalid_vid;
        }
      }
    } else if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.dst);
          }
        } else {
          if (IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.src);
          } else {
            e.src = invalid_vid;
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid load strategy.";
    }

    alive_ivnum_ = ivnum_;
    alive_ovnum_ = ovnum_;
    iv_alive_.init(ivnum_);
    ov_alive_.init(ovnum_);
    for (size_t i = 0; i < ivnum_; i++) {
      iv_alive_.set_bit(i);
    }
    for (size_t i = 0; i < ovnum_; i++) {
      ov_alive_.set_bit(i);
    }
    is_selfloops_.init(ivnum_);

    this->inner_vertices_.SetRange(0, ivnum_);
    this->outer_vertices_.SetRange(id_parser_.max_local_id() - ovnum_,
                                   id_parser_.max_local_id());
    this->vertices_.SetRange(0, ivnum_, id_parser_.max_local_id() - ovnum_,
                             id_parser_.max_local_id());
    initOuterVerticesOfFragment();

    buildCSR(edges, load_strategy_);

    ivdata_.clear();
    ivdata_.resize(ivnum_, dynamic::Value(rapidjson::kObjectType));
    ovdata_.clear();
    ovdata_.resize(ovnum_);
    if (sizeof(internal_vertex_t) > sizeof(vid_t)) {
      for (auto& v : vertices) {
        vid_t gid = v.vid;
        if (id_parser_.get_fragment_id(gid) == fid_) {
          ivdata_[id_parser_.get_local_id(gid)].Update(v.vdata);
        } else {
          auto iter = ovg2i_.find(gid);
          if (iter != ovg2i_.end()) {
            auto index = outerVertexLidToIndex(iter->second);
            ovdata_[index] = std::move(v.vdata);
          }
        }
      }
    }
  }

  void Init(fid_t fid, bool directed) {
    std::vector<internal_vertex_t> empty_vertices;
    std::vector<edge_t> empty_edges;
    Init(fid, directed, empty_vertices, empty_edges);
  }

  using base_t::Gid2Lid;
  using base_t::ie_;
  using base_t::oe_;
  using base_t::vm_ptr_;
  void Mutate(mutation_t& mutation) {
    vertex_t v;
    if (mutation.vertices_to_remove.empty() &&
        static_cast<double>(mutation.vertices_to_remove.size()) /
                static_cast<double>(this->GetVerticesNum()) <
            0.1) {
      std::set<vertex_t> sparse_set;
      for (auto gid : mutation.vertices_to_remove) {
        if (Gid2Vertex(gid, v)) {
          if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
            ie_.remove_vertex(v.GetValue());
          }
          oe_.remove_vertex(v.GetValue());
          sparse_set.insert(v);

          // remove vertex
          if (IsInnerVertex(v)) {
            iv_alive_.reset_bit(v.GetValue());
            --alive_ivnum_;
          } else {
            ov_alive_.reset_bit(outerVertexLidToIndex(v.GetValue()));
            --alive_ovnum_;
          }
        }
      }
      if (!sparse_set.empty()) {
        auto func = [&sparse_set](vid_t i, const nbr_t& e) {
          return sparse_set.find(e.neighbor) != sparse_set.end();
        };
        if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
          ie_.remove_if(func);
        }
        oe_.remove_if(func);
      }
    } else if (!mutation.vertices_to_remove.empty()) {
      grape::DenseVertexSet<vertices_t> dense_bitset(Vertices());
      for (auto gid : mutation.vertices_to_remove) {
        if (Gid2Vertex(gid, v)) {
          if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
            ie_.remove_vertex(v.GetValue());
          }
          oe_.remove_vertex(v.GetValue());
          dense_bitset.Insert(v);

          // remove vertex
          if (IsInnerVertex(v) && IsAliveInnerVertex(v)) {
            iv_alive_.reset_bit(v.GetValue());
            --alive_ivnum_;
            is_selfloops_.reset_bit(v.GetValue());
          } else {
            ov_alive_.reset_bit(outerVertexLidToIndex(v.GetValue()));
            --alive_ovnum_;
          }
        }
      }
      auto func = [&dense_bitset](vid_t i, const nbr_t& e) {
        return dense_bitset.Exist(e.neighbor);
      };
      if (!dense_bitset.Empty()) {
        if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
          ie_.remove_if(func);
        }
        oe_.remove_if(func);
      }
    }
    if (!mutation.edges_to_remove.empty()) {
      static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
      for (auto& e : mutation.edges_to_remove) {
        if (!(Gid2Lid(e.first, e.first) && Gid2Lid(e.second, e.second))) {
          e.first = sentinel;
        }
        if (e.first == e.second) {
          this->is_selfloops_.reset_bit(e.first);
        }
      }
      if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
        ie_.remove_reversed_edges(mutation.edges_to_remove);
      }
      oe_.remove_edges(mutation.edges_to_remove);
    }
    if (!mutation.edges_to_update.empty()) {
      static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
      for (auto& e : mutation.edges_to_update) {
        if (!(Gid2Lid(e.src, e.src) && Gid2Lid(e.dst, e.dst))) {
          e.src = sentinel;
        }
      }
      if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
        ie_.update_reversed_edges(mutation.edges_to_update);
      }
      oe_.update_edges(mutation.edges_to_update);
    }
    {
      vid_t ivnum = this->GetInnerVerticesNum();
      vid_t ovnum = this->GetOuterVerticesNum();
      auto& edges_to_add = mutation.edges_to_add;
      static constexpr vid_t invalid_vid = std::numeric_limits<vid_t>::max();
      for (auto& e : edges_to_add) {
        if (IsInnerVertexGid(e.src)) {
          e.src = id_parser_.get_local_id(e.src);
          if (IsInnerVertexGid(e.dst)) {
            e.dst = id_parser_.get_local_id(e.dst);
          } else {
            e.dst = parseOrAddOuterVertexGid(e.dst);
          }
        } else {
          if (IsInnerVertexGid(e.dst)) {
            e.src = parseOrAddOuterVertexGid(e.src);
            e.dst = id_parser_.get_local_id(e.dst);
          } else {
            e.src = invalid_vid;
          }
        }
      }
      vid_t new_ivnum = vm_ptr_->GetInnerVertexSize(fid_);
      vid_t new_ovnum = ovgid_.size();
      is_selfloops_.resize(new_ivnum);
      // reserve edges
      ie_.add_vertices(new_ivnum - ivnum, new_ovnum - ovnum);
      oe_.add_vertices(new_ivnum - ivnum, new_ovnum - ovnum);
      this->ivnum_ = new_ivnum;
      if (ovnum_ != new_ovnum) {
        ovnum_ = new_ovnum;
        initOuterVerticesOfFragment();
      }
      if (!edges_to_add.empty()) {
        if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
          oe_.reserve_forward_edges(edges_to_add);
          ie_.reserve_reversed_edges(edges_to_add);
        } else {
          oe_.reserve_edges(edges_to_add);
        }
        double rate = 0;
        if (directed_) {
          rate = static_cast<double>(edges_to_add.size()) /
                 static_cast<double>(oe_.edge_num());
        } else {
          rate = 2.0 * static_cast<double>(edges_to_add.size()) /
                 static_cast<double>(oe_.edge_num());
        }
        if (rate < oe_.dense_threshold) {
          addEdgesSparse(edges_to_add);
        } else {
          addEdgesDense(edges_to_add);
        }
      }

      this->inner_vertices_.SetRange(0, new_ivnum);
      this->outer_vertices_.SetRange(id_parser_.max_local_id() - new_ovnum,
                                     id_parser_.max_local_id());
      this->vertices_.SetRange(0, new_ivnum,
                               id_parser_.max_local_id() - new_ovnum,
                               id_parser_.max_local_id());
    }
    ivdata_.resize(this->ivnum_, dynamic::Value(rapidjson::kObjectType));
    ovdata_.resize(this->ovnum_);
    iv_alive_.resize(this->ivnum_);
    ov_alive_.resize(this->ovnum_);
    for (auto& v : mutation.vertices_to_add) {
      vid_t lid;
      if (IsInnerVertexGid(v.vid)) {
        this->InnerVertexGid2Lid(v.vid, lid);
        ivdata_[lid].Update(v.vdata);
        if (iv_alive_.get_bit(lid) == false) {
          iv_alive_.set_bit(lid);
          ++alive_ivnum_;
        }
      } else {
        if (this->OuterVertexGid2Lid(v.vid, lid)) {
          auto index = outerVertexLidToIndex(lid);
          ovdata_[index] = std::move(v.vdata);
          if (ov_alive_.get_bit(index) == false) {
            ov_alive_.set_bit(index);
            ++alive_ovnum_;
          }
        }
      }
    }
    for (auto& v : mutation.vertices_to_update) {
      vid_t lid;
      if (IsInnerVertexGid(v.vid)) {
        this->InnerVertexGid2Lid(v.vid, lid);
        ivdata_[lid] = std::move(v.vdata);
      } else {
        if (this->OuterVertexGid2Lid(v.vid, lid)) {
          ovdata_[outerVertexLidToIndex(lid)] = std::move(v.vdata);
        }
      }
    }
  }

  void PrepareToRunApp(const grape::CommSpec& comm_spec,
                       grape::PrepareConf conf) override {
    base_t::PrepareToRunApp(comm_spec, conf);
    if (conf.need_split_edges_by_fragment) {
      LOG(FATAL) << "MutableEdgecutFragment cannot split edges by fragment";
    } else if (conf.need_split_edges) {
      splitEdges();
    }
  }

  inline size_t GetEdgeNum() const override {
    return this->directed_ ? ie_.edge_num() + oe_.edge_num()
                           : oe_.edge_num() + is_selfloops_.count();
  }

  using base_t::InnerVertices;
  using base_t::IsInnerVertex;
  using base_t::OuterVertices;

  vid_t GetVerticesNum() const { return alive_ivnum_ + alive_ovnum_; }

  vid_t GetInnerVerticesNum() const { return alive_ivnum_; }

  vid_t GetOuterVerticesNum() const { return alive_ovnum_; }

  inline const vdata_t& GetData(const vertex_t& v) const override {
    return IsInnerVertex(v) ? ivdata_[v.GetValue()]
                            : ovdata_[outerVertexLidToIndex(v.GetValue())];
  }

  inline void SetData(const vertex_t& v, const vdata_t& val) override {
    if (IsInnerVertex(v)) {
      ivdata_[v.GetValue()] = val;
    } else {
      ovdata_[outerVertexLidToIndex(v.GetValue())] = val;
    }
  }

  bool OuterVertexGid2Lid(vid_t gid, vid_t& lid) const override {
    auto iter = ovg2i_.find(gid);
    if (iter != ovg2i_.end()) {
      lid = iter->second;
      return true;
    } else {
      return false;
    }
  }

  vid_t GetOuterVertexGid(vertex_t v) const override {
    return ovgid_[outerVertexLidToIndex(v.GetValue())];
  }

  bool IsOuterVertexGid(vid_t gid) const {
    return ovg2i_.find(gid) != ovg2i_.end();
  }

  inline bool Gid2Vertex(const vid_t& gid, vertex_t& v) const override {
    fid_t fid = id_parser_.get_fragment_id(gid);
    if (fid == fid_) {
      v.SetValue(id_parser_.get_local_id(gid));
      return true;
    } else {
      auto iter = ovg2i_.find(gid);
      if (iter != ovg2i_.end()) {
        v.SetValue(iter->second);
        return true;
      } else {
        return false;
      }
    }
  }

  inline vid_t Vertex2Gid(const vertex_t& v) const override {
    if (IsInnerVertex(v)) {
      return id_parser_.generate_global_id(fid_, v.GetValue());
    } else {
      return ovgid_[outerVertexLidToIndex(v.GetValue())];
    }
  }

  void ClearGraph(std::shared_ptr<vertex_map_t> vm_ptr) {
    vm_ptr_.reset();
    vm_ptr_ = vm_ptr;
    Init(fid_, directed_);
  }

  void ClearEdges() {
    if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
      ie_.clear_edges();
    }
    oe_.clear_edges();

    // clear outer_vertices map
    ovgid_.clear();
    ovg2i_.clear();
    ov_alive_.clear();
    this->ovnum_ = 0;
    this->alive_ovnum_ = 0;
    is_selfloops_.clear();
  }

  void CopyFrom(std::shared_ptr<DynamicFragment> source,
                const std::string& copy_type = "identical") {
    init(source->fid_, source->directed_);
    load_strategy_ = source->load_strategy_;
    copyVertices(source);

    // copy edges
    if (copy_type == "identical") {
      ie_.copy_from(source->ie_);
      oe_.copy_from(source->oe_);
    } else if (copy_type == "reverse") {
      assert(directed_);
      ie_.copy_from(source->oe_);
      oe_.copy_from(source->ie_);
    } else {
      LOG(FATAL) << "Unsupported copy type: " << copy_type;
    }
  }

  // generate directed graph from original undirected graph.
  void ToDirectedFrom(std::shared_ptr<DynamicFragment> source) {
    assert(!source->directed_);
    init(source->fid_, true);
    load_strategy_ = grape::LoadStrategy::kBothOutIn;
    copyVertices(source);

    ie_.copy_from(source->oe_);
    oe_.copy_from(source->oe_);
  }

  // generate undirected graph from original directed graph.
  void ToUndirectedFrom(std::shared_ptr<DynamicFragment> source) {
    assert(source->directed_);
    init(source->fid_, false);
    ie_.Init(0, 0);
    oe_.Init(0, 0);
    load_strategy_ = grape::LoadStrategy::kOnlyOut;
    copyVertices(source);
    ie_.add_vertices(ivnum_, ovnum_);
    oe_.add_vertices(ivnum_, ovnum_);

    mutation_t mutation;
    vid_t gid;
    for (auto& v : source->InnerVertices()) {
      gid = Vertex2Gid(v);
      for (const auto& e : source->GetOutgoingAdjList(v)) {
        mutation.edges_to_add.emplace_back(gid, Vertex2Gid(e.neighbor), e.data);
      }
      for (const auto& e : source->GetIncomingAdjList(v)) {
        if (IsOuterVertex(e.neighbor)) {
          mutation.edges_to_add.emplace_back(gid, Vertex2Gid(e.neighbor),
                                             e.data);
        }
      }
    }

    Mutate(mutation);
  }

  // induce a subgraph that contains the induced_vertices and the edges between
  // those vertices or a edge subgraph that contains the induced_edges and the
  // nodes incident to induced_edges.
  void InduceSubgraph(
      std::shared_ptr<DynamicFragment> source,
      const std::vector<oid_t>& induced_vertices,
      const std::vector<std::pair<oid_t, oid_t>>& induced_edges) {
    Init(source->fid_, source->directed_);

    mutation_t mutation;
    if (induced_edges.empty()) {
      induceFromVertices(source, induced_vertices, mutation.edges_to_add);
    } else {
      induceFromEdges(source, induced_edges, mutation.edges_to_add);
    }
    Mutate(mutation);
  }

  inline bool Oid2Gid(const oid_t& oid, vid_t& gid) const {
    return vm_ptr_->GetGid(oid, gid);
  }

  inline size_t selfloops_num() const { return is_selfloops_.count(); }

  inline bool HasNode(const oid_t& node) const {
    vid_t gid;
    return this->vm_ptr_->GetGid(fid_, node, gid) &&
           iv_alive_.get_bit(id_parser_.get_local_id(gid));
  }

  inline bool HasEdge(const oid_t& u, const oid_t& v) const {
    vid_t uid, vid;
    if (vm_ptr_->GetGid(u, uid) && vm_ptr_->GetGid(v, vid)) {
      vid_t ulid, vlid;
      if (IsInnerVertexGid(uid) && InnerVertexGid2Lid(uid, ulid) &&
          Gid2Lid(vid, vlid) && iv_alive_.get_bit(ulid)) {
        auto begin = oe_.get_begin(ulid);
        auto end = oe_.get_end(ulid);
        auto iter =
            grape::mutable_csr_impl::binary_search_one(begin, end, vlid);
        if (iter != end) {
          return true;
        }
      } else if (IsInnerVertexGid(vid) && InnerVertexGid2Lid(vid, vlid) &&
                 Gid2Lid(uid, ulid) && iv_alive_.get_bit(vlid)) {
        auto begin = directed_ ? ie_.get_begin(vlid) : oe_.get_begin(vlid);
        auto end = directed_ ? ie_.get_end(vlid) : oe_.get_end(vlid);
        auto iter =
            grape::mutable_csr_impl::binary_search_one(begin, end, ulid);
        if (iter != end) {
          return true;
        }
      }
    }
    return false;
  }

  inline bool GetEdgeData(const oid_t& u_oid, const oid_t& v_oid,
                          edata_t& data) const {
    vid_t uid, vid;
    if (vm_ptr_->GetGid(u_oid, uid) && vm_ptr_->GetGid(v_oid, vid)) {
      vid_t ulid, vlid;
      if (IsInnerVertexGid(uid) && InnerVertexGid2Lid(uid, ulid) &&
          Gid2Lid(vid, vlid) && iv_alive_.get_bit(ulid)) {
        auto begin = oe_.get_begin(ulid);
        auto end = oe_.get_end(ulid);
        auto iter =
            grape::mutable_csr_impl::binary_search_one(begin, end, vlid);
        if (iter != end) {
          data = iter->data;
          return true;
        }
      } else if (IsInnerVertexGid(vid) && InnerVertexGid2Lid(vid, vlid) &&
                 Gid2Lid(uid, ulid) && iv_alive_.get_bit(vlid)) {
        auto begin = directed_ ? ie_.get_begin(vlid) : oe_.get_begin(vlid);
        auto end = directed_ ? ie_.get_end(vlid) : oe_.get_end(vlid);
        auto iter =
            grape::mutable_csr_impl::binary_search_one(begin, end, ulid);
        if (iter != end) {
          data = iter->data;
          return true;
        }
      }
    }
    return false;
  }

  inline bool IsAliveInnerVertex(const vertex_t& v) const {
    return iv_alive_.get_bit(v.GetValue());
  }

  auto CollectPropertyKeysOnVertices()
      -> bl::result<std::map<std::string, dynamic::Type>> {
    std::map<std::string, dynamic::Type> prop_keys;

    for (const auto& v : InnerVertices()) {
      auto& data = ivdata_[v.GetValue()];

      for (auto member = data.MemberBegin(); member != data.MemberEnd();
           ++member) {
        std::string s_k = member->name.GetString();

        if (prop_keys.find(s_k) == prop_keys.end()) {
          prop_keys[s_k] = dynamic::GetType(member->value);
        } else {
          auto seen_type = prop_keys[s_k];
          auto curr_type = dynamic::GetType(member->value);

          if (seen_type != curr_type) {
            std::stringstream ss;
            ss << "OID: " << GetId(v) << " has key " << s_k << " with type "
               << curr_type << " but previous type is: " << seen_type;
            RETURN_GS_ERROR(vineyard::ErrorCode::kDataTypeError, ss.str());
          }
        }
      }
    }

    return prop_keys;
  }

  auto CollectPropertyKeysOnEdges()
      -> bl::result<std::map<std::string, dynamic::Type>> {
    std::map<std::string, dynamic::Type> prop_keys;

    auto extract_keys = [this, &prop_keys](
                            const vertex_t& u,
                            const adj_list_t& es) -> bl::result<void> {
      for (auto& e : es) {
        auto& data = e.data;

        for (auto member = data.MemberBegin(); member != data.MemberEnd();
             ++member) {
          std::string s_k = member->name.GetString();

          if (prop_keys.find(s_k) == prop_keys.end()) {
            prop_keys[s_k] = dynamic::GetType(member->value);
          } else {
            auto seen_type = prop_keys[s_k];
            auto curr_type = dynamic::GetType(member->value);

            if (seen_type != curr_type) {
              std::stringstream ss;
              ss << "Edge (OID): " << GetId(u) << " " << GetId(e.neighbor)
                 << " has key " << s_k << " with type " << curr_type
                 << " but previous type is: " << seen_type;
              RETURN_GS_ERROR(vineyard::ErrorCode::kDataTypeError, ss.str());
            }
          }
        }
      }
      return {};
    };

    for (const auto& v : InnerVertices()) {
      if (load_strategy_ == grape::LoadStrategy::kOnlyIn ||
          load_strategy_ == grape::LoadStrategy::kBothOutIn) {
        auto es = this->GetIncomingAdjList(v);
        if (es.NotEmpty()) {
          BOOST_LEAF_CHECK(extract_keys(v, es));
        }
      }

      if (load_strategy_ == grape::LoadStrategy::kOnlyOut ||
          load_strategy_ == grape::LoadStrategy::kBothOutIn) {
        auto es = this->GetOutgoingAdjList(v);
        if (es.NotEmpty()) {
          BOOST_LEAF_CHECK(extract_keys(v, es));
        }
      }
    }

    return prop_keys;
  }

  bl::result<dynamic::Type> GetOidType(const grape::CommSpec& comm_spec) const {
    auto oid_type = dynamic::Type::kNullType;
    if (this->alive_ivnum_ > 0) {
      // Get first alive vertex oid type.
      for (vid_t lid = 0; lid < ivnum_; ++lid) {
        if (iv_alive_.get_bit(lid)) {
          oid_t oid;
          vm_ptr_->GetOid(fid_, lid, oid);
          oid_type = dynamic::GetType(oid);
        }
      }
    }
    grape::Communicator comm;
    dynamic::Type max_type;
    comm.InitCommunicator(comm_spec.comm());
    comm.Max(oid_type, max_type);

    if (max_type != dynamic::Type::kInt64Type &&
        max_type != dynamic::Type::kDoubleType &&
        max_type != dynamic::Type::kStringType &&
        max_type != dynamic::Type::kNullType) {
      LOG(FATAL) << "Unsupported oid type.";
    }
    return max_type;
  }

 public:
  using base_t::GetOutgoingAdjList;
  inline adj_list_t GetIncomingAdjList(const vertex_t& v) override {
    if (!this->directed_) {
      return adj_list_t(oe_.get_begin(v.GetValue()), oe_.get_end(v.GetValue()));
    }
    return adj_list_t(ie_.get_begin(v.GetValue()), ie_.get_end(v.GetValue()));
  }

  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v) const override {
    if (!this->directed_) {
      return const_adj_list_t(oe_.get_begin(v.GetValue()),
                              oe_.get_end(v.GetValue()));
    }
    return const_adj_list_t(ie_.get_begin(v.GetValue()),
                            ie_.get_end(v.GetValue()));
  }

  fragment_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                         fid_t dst_fid) override {
    return fragment_adj_list_t(
        get_oe_begin(v), get_oe_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

  fragment_const_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                               fid_t dst_fid) const override {
    return fragment_const_adj_list_t(
        get_oe_begin(v), get_oe_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

  fragment_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                         fid_t dst_fid) override {
    if (!this->directed_) {
      return fragment_adj_list_t(
          get_oe_begin(v), get_oe_end(v), [this, dst_fid](const nbr_t& nbr) {
            return this->GetFragId(nbr.get_neighbor()) == dst_fid;
          });
    }
    return fragment_adj_list_t(
        get_ie_begin(v), get_ie_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

  fragment_const_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                               fid_t dst_fid) const override {
    if (!this->directed_) {
      return fragment_const_adj_list_t(
          get_oe_begin(v), get_oe_end(v), [this, dst_fid](const nbr_t& nbr) {
            return this->GetFragId(nbr.get_neighbor()) == dst_fid;
          });
    }
    return fragment_const_adj_list_t(
        get_ie_begin(v), get_ie_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

 public:
  using base_t::get_ie_begin;
  using base_t::get_ie_end;
  using base_t::get_oe_begin;
  using base_t::get_oe_end;

 public:
  using adj_list_t = typename base_t::adj_list_t;
  using const_adj_list_t = typename base_t::const_adj_list_t;
  inline adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(get_ie_begin(v), iespliter_[v]);
  }

  inline const_adj_list_t GetIncomingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(get_ie_begin(v), iespliter_[v]);
  }

  inline adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(iespliter_[v], get_ie_end(v));
  }

  inline const_adj_list_t GetIncomingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(iespliter_[v], get_ie_end(v));
  }

  inline adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(get_oe_begin(v), oespliter_[v]);
  }

  inline const_adj_list_t GetOutgoingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(get_oe_begin(v), oespliter_[v]);
  }

  inline adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(oespliter_[v], get_oe_end(v));
  }

  inline const_adj_list_t GetOutgoingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(oespliter_[v], get_oe_end(v));
  }

 private:
  inline vid_t outerVertexLidToIndex(vid_t lid) const {
    return id_parser_.max_local_id() - lid - 1;
  }

  inline vid_t outerVertexIndexToLid(vid_t index) const {
    return id_parser_.max_local_id() - index - 1;
  }

  void splitEdges() {
    auto& inner_vertices = InnerVertices();
    iespliter_.Init(inner_vertices);
    oespliter_.Init(inner_vertices);
    int inner_neighbor_count = 0;
    for (auto& v : inner_vertices) {
      inner_neighbor_count = 0;
      auto ie = GetIncomingAdjList(v);
      for (auto& e : ie) {
        if (IsInnerVertex(e.neighbor)) {
          ++inner_neighbor_count;
        }
      }
      iespliter_[v] = get_ie_begin(v) + inner_neighbor_count;

      inner_neighbor_count = 0;
      auto oe = GetOutgoingAdjList(v);
      for (auto& e : oe) {
        if (IsInnerVertex(e.neighbor)) {
          ++inner_neighbor_count;
        }
      }
      oespliter_[v] = get_oe_begin(v) + inner_neighbor_count;
    }
  }

  vid_t parseOrAddOuterVertexGid(vid_t gid) {
    auto iter = ovg2i_.find(gid);
    if (iter != ovg2i_.end()) {
      return iter->second;
    } else {
      ++ovnum_;
      vid_t lid = id_parser_.max_local_id() - ovnum_;
      ovgid_.push_back(gid);
      ovg2i_.emplace(gid, lid);
      return lid;
    }
  }

  void initOuterVerticesOfFragment() {
    outer_vertices_of_frag_.resize(fnum_);
    for (auto& vec : outer_vertices_of_frag_) {
      vec.clear();
    }
    for (vid_t i = 0; i < ovnum_; ++i) {
      fid_t fid = id_parser_.get_fragment_id(ovgid_[i]);
      outer_vertices_of_frag_[fid].push_back(
          vertex_t(outerVertexIndexToLid(i)));
    }
  }

  // Return true if add a new edge, otherwise false.
  bool updateOrAddEdgeOut(const edge_t& e) {
    bool ret = false;  // assume it just update existed edge.
    if (e.src < ivnum_) {
      auto begin = oe_.get_begin(e.src);
      auto end = oe_.get_end(e.src);
      if (begin == end) {
        oe_.add_edge(e);
        ret = true;
      } else {
        auto iter = std::find_if(begin, end, [&e](const auto& val) {
          return val.neighbor.GetValue() == e.dst;
        });
        if (iter != end) {
          iter->data.Update(e.edata);
        } else {
          oe_.add_edge(e);
          ret = true;
        }
      }
      if (ret && e.src == e.dst) {
        is_selfloops_.set_bit(e.src);
        return ret;
      }
    }

    if (e.dst < ivnum_) {
      auto begin = oe_.get_begin(e.dst);
      auto end = oe_.get_end(e.dst);
      if (begin == end) {
        oe_.add_reversed_edge(e);
        ret = true;
      } else {
        auto iter = std::find_if(begin, end, [&e](const auto& val) {
          return val.neighbor.GetValue() == e.src;
        });
        if (iter != end) {
          iter->data.Update(e.edata);
        } else {
          oe_.add_reversed_edge(e);
          ret = true;
        }
      }
    }
    return ret;
  }

  // Return true if add a new edge, otherwise false.
  bool updateOrAddEdgeOutIn(const edge_t& e) {
    bool ret = false;  // assume it just update existed edge.
    if (e.src < ivnum_) {
      // src is inner vertex.
      auto begin = oe_.get_begin(e.src);
      auto end = oe_.get_end(e.src);
      if (begin == end) {
        oe_.add_edge(e);
        ret = true;
      } else {
        auto iter = std::find_if(begin, end, [&e](const auto& val) {
          return val.neighbor.GetValue() == e.dst;
        });
        if (iter != end) {
          // edge existed, update it.
          iter->data.Update(e.edata);
        } else {
          oe_.add_edge(e);
          ret = true;
        }
      }
      if (ret && e.src == e.dst) {
        is_selfloops_.set_bit(e.src);
      }
    }

    if (e.dst < ivnum_) {
      // dst is the inner vertex;
      auto begin = ie_.get_begin(e.dst);
      auto end = ie_.get_end(e.dst);
      if (begin == end) {
        ie_.add_reversed_edge(e);
        ret = true;
      } else {
        auto iter = std::find_if(begin, end, [&e](const auto& val) {
          return val.neighbor.GetValue() == e.src;
        });
        if (iter != end) {
          iter->data.Update(e.edata);
        } else {
          ie_.add_reversed_edge(e);
          ret = true;
        }
      }
    }
    return ret;
  }

  void addEdgesDense(std::vector<edge_t>& edges) {
    if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
      std::vector<int> oe_head_degree_to_add(oe_.head_vertex_num(), 0),
          ie_head_degree_to_add(ie_.head_vertex_num(), 0), dummy_tail_degree;
      for (auto& e : edges) {
        if (updateOrAddEdgeOutIn(e)) {
          if (e.src < ivnum_) {
            ++oe_head_degree_to_add[oe_.head_index(e.src)];
          }
          if (e.dst < ivnum_) {
            ++ie_head_degree_to_add[ie_.head_index(e.dst)];
          }
        }
      }
      oe_.dedup_or_sort_neighbors_dense(oe_head_degree_to_add,
                                        dummy_tail_degree);
      ie_.dedup_or_sort_neighbors_dense(ie_head_degree_to_add,
                                        dummy_tail_degree);
    } else {
      std::vector<int> oe_head_degree_to_add(ivnum_, 0), dummy_tail_degree;
      for (auto& e : edges) {
        if (updateOrAddEdgeOut(e)) {
          if (e.src < ivnum_) {
            ++oe_head_degree_to_add[oe_.head_index(e.src)];
          }
          if (e.dst < ivnum_ && e.src != e.dst) {
            ++oe_head_degree_to_add[oe_.head_index(e.dst)];
          }
        }
      }
      oe_.dedup_or_sort_neighbors_dense(oe_head_degree_to_add,
                                        dummy_tail_degree);
    }
  }

  void addEdgesSparse(std::vector<edge_t>& edges) {
    if (load_strategy_ == grape::LoadStrategy::kBothOutIn) {
      std::map<vid_t, int> oe_head_degree_to_add, ie_head_degree_to_add,
          dummy_tail_degree;
      for (auto& e : edges) {
        if (updateOrAddEdgeOutIn(e)) {
          if (e.src < ivnum_) {
            ++oe_head_degree_to_add[oe_.head_index(e.src)];
          }
          if (e.dst < ivnum_) {
            ++ie_head_degree_to_add[ie_.head_index(e.dst)];
          }
        }
      }
      oe_.dedup_or_sort_neighbors_sparse(oe_head_degree_to_add,
                                         dummy_tail_degree);
      ie_.dedup_or_sort_neighbors_sparse(ie_head_degree_to_add,
                                         dummy_tail_degree);
    } else {
      std::map<vid_t, int> oe_head_degree_to_add, dummy_tail_degree;
      for (auto& e : edges) {
        if (updateOrAddEdgeOut(e)) {
          if (e.src < ivnum_) {
            ++oe_head_degree_to_add[oe_.head_index(e.src)];
          }
          if (e.dst < ivnum_ && e.src != e.dst) {
            ++oe_head_degree_to_add[oe_.head_index(e.dst)];
          }
        }
      }
      oe_.dedup_or_sort_neighbors_sparse(oe_head_degree_to_add,
                                         dummy_tail_degree);
    }
  }

  void copyVertices(std::shared_ptr<DynamicFragment>& source) {
    this->ivnum_ = source->ivnum_;
    this->ovnum_ = source->ovnum_;
    this->alive_ivnum_ = source->alive_ivnum_;
    this->alive_ovnum_ = source->alive_ovnum_;
    this->fnum_ = source->fnum_;
    this->iv_alive_.copy(source->iv_alive_);
    this->ov_alive_.copy(source->ov_alive_);
    this->is_selfloops_.copy(source->is_selfloops_);

    ovg2i_ = source->ovg2i_;
    ovgid_.resize(ovnum_);
    memcpy(&ovgid_[0], &(source->ovgid_[0]), ovnum_ * sizeof(vid_t));

    ivdata_.clear();
    ivdata_.resize(ivnum_);
    for (size_t i = 0; i < ivnum_; ++i) {
      ivdata_[i] = source->ivdata_[i];
    }

    this->inner_vertices_.SetRange(0, ivnum_);
    this->outer_vertices_.SetRange(id_parser_.max_local_id() - ovnum_,
                                   id_parser_.max_local_id());
    this->vertices_.SetRange(0, ivnum_, id_parser_.max_local_id() - ovnum_,
                             id_parser_.max_local_id());
  }

  // induce subgraph from induced_nodes
  void induceFromVertices(std::shared_ptr<DynamicFragment>& source,
                          const std::vector<oid_t>& induced_vertices,
                          std::vector<edge_t>& edges) {
    vertex_t vertex;
    vid_t gid, dst_gid;
    for (const auto& oid : induced_vertices) {
      if (source->GetVertex(oid, vertex)) {
        if (source->IsInnerVertex(vertex)) {
          // store the vertex data
          CHECK(vm_ptr_->GetGid(fid_, oid, gid));
          auto lid = id_parser_.get_local_id(gid);
          ivdata_[lid] = source->GetData(vertex);
        } else {
          continue;  // ignore outer vertex.
        }

        for (const auto& e : source->GetOutgoingAdjList(vertex)) {
          auto dst_oid = source->GetId(e.get_neighbor());
          if (std::find(induced_vertices.begin(), induced_vertices.end(),
                        dst_oid) != induced_vertices.end()) {
            CHECK(Oid2Gid(dst_oid, dst_gid));
            edges.emplace_back(gid, dst_gid, e.get_data());
          }
        }
        if (directed_) {
          // filter the cross-fragment incoming edges
          for (const auto& e : source->GetIncomingAdjList(vertex)) {
            if (source->IsOuterVertex(e.get_neighbor())) {
              auto dst_oid = source->GetId(e.get_neighbor());
              if (std::find(induced_vertices.begin(), induced_vertices.end(),
                            dst_oid) != induced_vertices.end()) {
                CHECK(Oid2Gid(dst_oid, dst_gid));
                edges.emplace_back(dst_gid, gid, e.get_data());
              }
            }
          }
        }
      }
    }
  }

  // induce edge_subgraph from induced_edges
  void induceFromEdges(
      std::shared_ptr<DynamicFragment>& source,
      const std::vector<std::pair<oid_t, oid_t>>& induced_edges,
      std::vector<edge_t>& edges) {
    vertex_t vertex;
    vid_t gid, dst_gid;
    edata_t edata;
    for (auto& e : induced_edges) {
      const auto& src_oid = e.first;
      const auto& dst_oid = e.second;
      if (source->HasEdge(src_oid, dst_oid)) {
        if (vm_ptr_->GetGid(fid_, src_oid, gid)) {
          // src is inner vertex
          auto lid = id_parser_.get_local_id(gid);
          CHECK(source->GetVertex(src_oid, vertex));
          ivdata_[lid] = source->GetData(vertex);
          CHECK(vm_ptr_->GetGid(dst_oid, dst_gid));
          CHECK(source->GetEdgeData(src_oid, dst_oid, edata));
          edges.emplace_back(gid, dst_gid, edata);
          if (gid != dst_gid && id_parser_.get_fragment_id(dst_gid) == fid_) {
            // dst is inner vertex too
            CHECK(source->GetVertex(dst_oid, vertex));
            ivdata_[id_parser_.get_local_id(dst_gid)] = source->GetData(vertex);
          }
        } else if (vm_ptr_->GetGid(fid_, dst_oid, dst_gid)) {
          // dst is inner vertex but src is outer vertex
          CHECK(source->GetVertex(dst_oid, vertex));
          ivdata_[id_parser_.get_local_id(dst_gid)] = source->GetData(vertex);
          CHECK(vm_ptr_->GetGid(src_oid, gid));
          source->GetEdgeData(src_oid, dst_oid, edata);
          if (directed_) {
            edges.emplace_back(gid, dst_gid, edata);
          } else {
            edges.emplace_back(dst_gid, gid, edata);
          }
        }
      }
    }
  }

  using base_t::ivnum_;
  vid_t ovnum_;
  vid_t alive_ivnum_, alive_ovnum_;
  using base_t::directed_;
  using base_t::fid_;
  using base_t::fnum_;
  using base_t::id_parser_;
  grape::LoadStrategy load_strategy_;

  ska::flat_hash_map<vid_t, vid_t> ovg2i_;
  std::vector<vid_t> ovgid_;
  grape::Array<vdata_t, grape::Allocator<vdata_t>> ivdata_;
  grape::Array<vdata_t, grape::Allocator<vdata_t>> ovdata_;
  grape::Bitset iv_alive_;
  grape::Bitset ov_alive_;
  grape::Bitset is_selfloops_;

  grape::VertexArray<inner_vertices_t, nbr_t*> iespliter_, oespliter_;

  using base_t::outer_vertices_of_frag_;

  template <typename _vdata_t, typename _edata_t>
  friend class DynamicProjectedFragment;
};

class DynamicFragmentMutator {
  using fragment_t = DynamicFragment;
  using vertex_map_t = typename fragment_t::vertex_map_t;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;
  using mutation_t = typename fragment_t::mutation_t;
  using partitioner_t = typename vertex_map_t::partitioner_t;

 public:
  explicit DynamicFragmentMutator(const grape::CommSpec& comm_spec,
                                  std::shared_ptr<fragment_t> fragment)
      : comm_spec_(comm_spec),
        fragment_(fragment),
        vm_ptr_(fragment->GetVertexMap()) {
    comm_spec_.Dup();
  }

  ~DynamicFragmentMutator() = default;

  void ModifyVertices(dynamic::Value& vertices_to_modify,
                      const dynamic::Value& common_attrs,
                      const rpc::ModifyType& modify_type) {
    mutation_t mutation;
    auto& partitioner = vm_ptr_->GetPartitioner();
    oid_t oid;
    vid_t gid;
    vdata_t v_data;
    fid_t v_fid, fid = fragment_->fid();
    for (auto& v : vertices_to_modify) {
      v_data = common_attrs;
      // v could be [id, attrs] or id
      if (v.IsArray() && v.Size() == 2 && v[1].IsObject()) {
        oid = std::move(v[0]);
        v_data.Update(vdata_t(v[1]));
      } else {
        oid = std::move(v);
      }
      v_fid = partitioner.GetPartitionId(oid);
      if (modify_type == rpc::NX_ADD_NODES) {
        vm_ptr_->AddVertex(oid, gid);
        if (v_fid == fid) {
          mutation.vertices_to_add.emplace_back(gid, std::move(v_data));
        }
      } else {
        // UPDATE or DELETE, if not exist the node, continue.
        if (!vm_ptr_->GetGid(v_fid, oid, gid)) {
          continue;
        }
      }
      if (modify_type == rpc::NX_UPDATE_NODES && v_fid == fid) {
        mutation.vertices_to_update.emplace_back(gid, std::move(v_data));
      }
      if (modify_type == rpc::NX_DEL_NODES &&
          (v_fid == fid || fragment_->IsOuterVertexGid(gid))) {
        mutation.vertices_to_remove.emplace_back(gid);
      }
    }
    fragment_->Mutate(mutation);
  }

  void ModifyEdges(dynamic::Value& edges_to_modify,
                   const dynamic::Value& common_attrs,
                   const rpc::ModifyType modify_type,
                   const std::string weight) {
    edata_t e_data;
    oid_t src, dst;
    vid_t src_gid, dst_gid;
    fid_t src_fid, dst_fid, fid = fragment_->fid();
    auto& partitioner = vm_ptr_->GetPartitioner();
    mutation_t mutation;
    mutation.edges_to_add.reserve(edges_to_modify.Size());
    mutation.vertices_to_add.reserve(edges_to_modify.Size() * 2);
    for (auto& e : edges_to_modify) {
      // the edge could be [src, dst] or [srs, dst, value] or [src, dst,
      // {"key": val}]
      e_data = common_attrs;
      if (e.Size() == 3) {
        if (weight.empty()) {
          e_data.Update(edata_t(e[2]));
        } else {
          e_data.Insert(weight, edata_t(e[2]));
        }
      }
      src = std::move(e[0]);
      dst = std::move(e[1]);
      src_fid = partitioner.GetPartitionId(src);
      dst_fid = partitioner.GetPartitionId(dst);
      if (modify_type == rpc::NX_ADD_EDGES) {
        bool src_added = vm_ptr_->AddVertex(src, src_gid);
        bool dst_added = vm_ptr_->AddVertex(dst, dst_gid);
        if (src_fid == fid && src_added) {
          vdata_t empty_data(rapidjson::kObjectType);
          mutation.vertices_to_add.emplace_back(src_gid, std::move(empty_data));
        }
        if (dst_fid == fid && dst_added) {
          vdata_t empty_data(rapidjson::kObjectType);
          mutation.vertices_to_add.emplace_back(dst_gid, std::move(empty_data));
        }
      } else {
        if (!vm_ptr_->GetGid(src_fid, src, src_gid) ||
            !vm_ptr_->GetGid(dst_fid, dst, dst_gid)) {
          continue;
        }
      }
      if (modify_type == rpc::NX_ADD_EDGES) {
        if (src_fid == fid || dst_fid == fid) {
          mutation.edges_to_add.emplace_back(src_gid, dst_gid,
                                             std::move(e_data));
        }
      } else if (modify_type == rpc::NX_DEL_EDGES) {
        if (src_fid == fid || dst_fid == fid) {
          mutation.edges_to_remove.emplace_back(src_gid, dst_gid);
          if (!fragment_->directed() && src_gid != dst_gid) {
            mutation.edges_to_remove.emplace_back(dst_gid, src_gid);
          }
        }
      } else if (modify_type == rpc::NX_UPDATE_EDGES) {
        if (src_fid == fid || dst_fid == fid) {
          mutation.edges_to_update.emplace_back(src_gid, dst_gid, e_data);
          if (!fragment_->directed()) {
            mutation.edges_to_update.emplace_back(dst_gid, src_gid, e_data);
          }
        }
      }
    }
    fragment_->Mutate(mutation);
  }

 private:
  grape::CommSpec comm_spec_;
  std::shared_ptr<fragment_t> fragment_;
  std::shared_ptr<vertex_map_t> vm_ptr_;
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_CORE_FRAGMENT_DYNAMIC_FRAGMENT_H_
