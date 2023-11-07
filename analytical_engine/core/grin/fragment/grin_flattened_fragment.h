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

#ifndef ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_GRIN_H_
#define ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_GRIN_H_

#include <arrow/util/iterator.h>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/grin/fragment/grin_util.h" 

namespace grape {
class CommSpec;
}

namespace gs {

/**
 * @brief This class represents the fragment flattened from ArrowFragment.
 * Different from ArrowProjectedFragment, an ArrowFlattenedFragment derives from
 * an ArrowFragment, but flattens all the labels to one type, result in a graph
 * with a single type of vertices and a single type of edges. Optionally,
 * a common property across labels of vertices(reps., edges) in the
 * ArrowFragment will be reserved as vdata(resp, edata).
 * ArrowFlattenedFragment usually used as a wrapper for ArrowFragment to run the
 * applications/algorithms defined in NetworkX or Analytical engine,
 * since these algorithms need the topology of the whole (property) graph.
 *
 * @tparam OID_T
 * @tparam VID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class GRINFlattenedFragment {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using gid_t = int64_t;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = grin_util::Vertex;
  using internal_id_t = int64_t;
  using fid_t = GRIN_PARTITION_ID;
  using vertex_range_t = grin_util::VertexRange;
  using inner_vertices_t = vertex_range_t;
  using outer_vertices_t = vertex_range_t;
  using vertices_t = vertex_range_t;

  using adj_list_t =
      grin_util::AdjList<EDATA_T>;

  template <typename DATA_T>
  using vertex_array_t = grin_util::VertexArray<DATA_T>;

  template <typename DATA_T>
  using inner_vertex_array_t = grin_util::VertexArray<DATA_T>;

  template <typename DATA_T>
  using outer_vertex_array_t = grin_util::VertexArray<DATA_T>;

  // This member is used by grape::check_load_strategy_compatible()
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  GRINFlattenedFragment() = default;

#if defined(GRIN_WITH_VERTEX_PROPERTY) && defined(GRIN_WITH_EDGE_PROPERTY)  // property graph storage
  explicit GRINFlattenedFragment(GRIN_PARTITIONED_GRAPH partitioned_graph, GRIN_PARTITION partition,
      const std::string& v_prop_name, const std::string& e_prop_name)
        : pg_(partitioned_graph), partition_(partition), v_prop_(v_prop_name), e_prop_(e_prop_name) {
    g_ = grin_get_local_graph_by_partition(pg_, partition_);
    tvnum_ = ivnum_ = ovnum_ = 0;
    vtl_ = grin_get_vertex_type_list(g_);
    etl_ = grin_get_edge_type_list(g_);
    auto vt = grin_get_vertex_type_from_list(g_, vtl_, 0);
    vp_ = grin_get_vertex_property_by_name(g_, vt, v_prop_name.c_str());
    tvl_ = grin_get_vertex_list_by_type(g_, vt);
    ivl_ = grin_get_vertex_list_by_type_select_master(g_, vt);
    ovl_ = grin_get_vertex_list_by_type_select_mirror(g_, vt);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    tvnum_ = grin_get_vertex_list_size(g_, tvl_);
    ivnum_ = grin_get_vertex_list_size(g_, ivl_);
    ovnum_ = grin_get_vertex_list_size(g_, ovl_);

    auto et = grin_get_edge_type_from_list(g_, etl_, 0);
    auto inner_vertices = this->InnerVertices();
    for (const auto& v : inner_vertices) {
      auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
      v2iadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v.grin_v, et);
      v2oadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v.grin_v, et);
    }
#elif defined(GRIN_ENABLE_VERTEX_LIST_ITERATOR)
    auto iv_iter = grin_get_vertex_list_begin(g_, ivl_);
    auto et = grin_get_edge_type_from_list(g_, etl_, 0);
    while (!grin_is_vertex_list_end(g_, iv_iter)) {
        auto v = grin_get_vertex_from_iter(g_, iv_iter);
        auto internal_id = grin_get_vertex_internal_id(g_, v);
        ++ivnum_;
        v2iadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v, et);
        v2oadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v, et);
        grin_destroy_vertex(g_, v);
        grin_get_next_vertex_list_iter(g_, iv_iter);
    }
    grin_destroy_vertex_list_iter(g_, iv_iter);
    while (!grin_is_vertex_list_end(g_, ov_iter)) {
      ++ovnum_;
      grin_get_next_vertex_list_iter(g_, ov_iter);
    }
    grin_destroy_vertex_list_iter(g_, ov_iter);
    tvnum_ = ivnum_ + ovnum_;
#endif

    // check if the vertex property and edge property are valid
    /*
    if (!v_prop_name.empty()) {
      auto properties = grin_get_vertex_properties_by_name(g_, v_prop_name.c_str());
      auto sz = grin_get_vertex_property_list_size(g_, properties);
      if (sz == 0) {
        LOG(FATAL) << "The vertex property " << v_prop_name << " is not found.";
      }
      grin_destroy_vertex_property_list(g_, properties);
    }
    if (!e_prop_name.empty()) {
      auto properties = grin_get_edge_properties_by_name(g_, e_prop_name.c_str());
      auto sz = grin_get_edge_property_list_size(g_, properties);
      if (sz == 0) {
        LOG(FATAL) << "The edge property " << e_prop_name << " is not found.";
      }
      grin_destroy_edge_property_list(g_, properties);
    }
    */

    // Initialize fid and fnum
#if defined(GRIN_ENABLE_GRAPH_PARTITION) && defined(GRIN_TRAIT_NATURAL_ID_FOR_PARTITION)
    fid_ = grin_get_partition_id(pg_, partition_);
#elif !defined(GRIN_ENABLE_GRAPH_PARTITION)
    fid_ = 0;
#endif

#ifdef GRIN_ENABLE_GRAPH_PARTITION
    fnum_ = grin_get_total_partitions_number(pg_);
#else
    fnum_ = 1;
#endif 
  }
#endif // GRIN_WITH_VERTEX_PROPERTY && GRIN_WITH_EDGE_PROPERTY

  ~GRINFlattenedFragment() {
    grin_destroy_vertex_list(g_, tvl_);
    grin_destroy_vertex_list(g_, ivl_);
    grin_destroy_vertex_list(g_, ovl_);
    for (const auto& pair : v2iadj_) {
      grin_destroy_adjacent_list(g_, pair.second);
    }
    for (const auto& pair : v2oadj_) {
      grin_destroy_adjacent_list(g_, pair.second);
    }
    grin_destroy_vertex_type_list(g_, vtl_);
    grin_destroy_edge_type_list(g_, etl_);
    grin_destroy_graph(g_);
  }

  inline fid_t fid() const {
    return fid_;
  }

  inline fid_t fnum() const {
    return fnum_;
  }

  inline bool directed() const { return grin_is_directed(g_); }

  inline vertex_range_t Vertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, tvl_, 0, tvnum_);
#else
    return vertex_range_t(g_, tvl_, 0);
#endif
  }

  inline vertex_range_t InnerVertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, ivl_, 0, ivnum_);
#else
    return vertex_range_t(g_, ivl_, 0);
#endif
  }

  inline vertex_range_t OuterVertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, ovl_, 0, ovnum_);
#else
    return vertex_range_t(g_, ovl_, 0);
#endif
  }

  inline oid_t GetId(const vertex_t& v) const { // TODO(wanglei)
    // return v.grin_v;
    return grin_get_vertex_external_id_of_int64(g_, v.grin_v);
  }

bool GetVertex(gid_t& ref, vertex_t& v) const {
    auto v_ref = grin_deserialize_int64_to_vertex_ref(g_, ref);
    auto grin_v = grin_get_vertex_from_vertex_ref(g_, v_ref);
    if (grin_v == GRIN_NULL_VERTEX) {
      grin_destroy_vertex(g_, v_ref);
      grin_destroy_vertex_ref(g_, v_ref);
      return false;
    }
    v.Refresh(g_, grin_v);
    if (IsInnerVertex(v)) {
      return true;
    }
    return false;
  }

#ifdef GRIN_ENABLE_VERTEX_ORIGINAL_ID_OF_INT64
  bool GetVertex(oid_t& oid, vertex_t& v) const {
    auto grin_v = grin_get_vertex_by_external_id_of_int64(g_, oid);
    if (grin_v == GRIN_NULL_VERTEX) {
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  bool GetInnerVertex(oid_t& oid, vertex_t& v) const {
    auto grin_v = grin_get_vertex_by_external_id_of_int64(g_, oid);
    if (grin_v == GRIN_NULL_VERTEX || grin_is_mirror_vertex(g_, grin_v)) {
      grin_destroy_vertex(g_, grin_v);
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  inline bool GetId(const vertex_t& v, oid_t& oid) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    oid = grin_get_vertex_external_id_of_int64(g_, v.grin_v);
    return true;
  }

  inline oid_t GetId(const vertex_t& v) const {
    return grin_get_vertex_external_id_of_int64(g_, v.grin_v);
  }

/*
  inline oid_t Gid2Oid(const gid_t& gid) const {
    return fragment_->Gid2Oid(gid);
  }

  inline bool Oid2Gid(const oid_t& oid, gid_t& gid) const {
    for (label_id_t label = 0; label < fragment_->vertex_label_num(); label++) {
      if (fragment_->Oid2Gid(label, oid, gid)) {
        return true;
      }
    }
    return false;
  }
*/
#endif

  inline GRIN_PARTITION_ID GetFragId(const vertex_t& u) const {
    auto v_ref = grin_get_vertex_ref_by_vertex(g_, u.grin_v);
    auto partition = grin_get_master_partition_from_vertex_ref(g_, v_ref);
    auto fid = grin_get_partition_id(pg_, partition);
    grin_destroy_vertex_ref(g_, v_ref);
    grin_destroy_partition(g_, partition);
    return fid;
  }

  inline bool Gid2Vertex(const gid_t& ref, vertex_t& v) const {
    auto v_ref = grin_deserialize_int64_to_vertex_ref(g_, ref);
    auto grin_v = grin_get_vertex_from_vertex_ref(g_, v_ref);
    if (grin_v == GRIN_NULL_VERTEX) {
      grin_destroy_vertex(g_, v_ref);
      grin_destroy_vertex_ref(g_, v_ref);
      return false;
    }
    grin_destroy_vertex_ref(g_, v_ref);
    v.Refresh(g_, grin_v);
    return true;
  }

  inline gid_t GetInnerVertexGid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline gid_t GetOuterVertexGid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline const gid_t Vertex2Gid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  bool GetData(const vertex_t& v, vdata_t& value) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    auto vtype = grin_get_vertex_type(g_, v.grin_v);
    auto v_prop = grin_get_vertex_property_by_name(g_, vtype, v_prop_.data());
    bool succ = false;
    if (v_prop != GRIN_NULL_VERTEX_PROPERTY) {
      value = grin_get_vertex_property_value_of_int64(g_, v.grin_v, v_prop);
      succ = true;
    }
    grin_destroy_vertex_type(g_, vtype);
    grin_destroy_vertex_property(g_, v_prop);
    return succ;
  }

  inline size_t GetInnerVerticesNum() const { return ivnum_; }

  inline size_t GetOuterVerticesNum() const { return ovnum_; }

  inline size_t GetVerticesNum() const { return tvnum_; }

  inline size_t GetEdgeNum() const {
#ifdef GRIN_WITH_EDGE_PROPERTY
    auto et = grin_get_edge_type_from_list(g_, etl_, 0);
    return grin_get_edge_num_by_type(g_, et);
#else
    return grin_get_edge_num(g_, );
#endif
  }

  inline bool IsInnerVertex(const vertex_t& v) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    return grin_is_master_vertex(g_, v.grin_v);
  }

  inline bool IsOuterVertex(const vertex_t& v) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    return grin_is_mirror_vertex(g_, v.grin_v);
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto al = v2oadj_.at(internal_id);
    // auto al = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v.grin_v, 0);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto sz = grin_get_adjacent_list_size(g_, al);
    return adj_list_t(g_, al, e_prop_.c_str(), 0, sz);
#else
    return adj_list_t(g_, al, e_prop_.c_str());
#endif
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto al = v2iadj_.at(internal_id);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    // auto _al = static_cast<GRIN_ADJACENT_LIST_T*>(al);
    // auto sz =  _al->offsets[_al->etype_end - _al->etype_begin];
    auto sz = grin_get_adjacent_list_size(g_, al);
    // return adj_list_t(g_, al, e_prop_.c_str(), 0, sz);
    return adj_list_t(g_, al, e_prop_.c_str(), 0, sz);
#else
    return adj_list_t(g_, al, e_prop_.c_str());
#endif
  }

  inline int GetLocalOutDegree(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto al = v2oadj_.at(internal_id);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return static_cast<int>(grin_get_adjacent_list_size(g_, al));
#else
    int degree = 0;
    auto iter = grin_get_adjacent_list_begin(g_, al);
    while (!grin_is_adjacent_list_end(g_, iter)) {
      ++degree;
      grin_get_next_adjacent_list_iter(g_, iter);
    }
    grin_destroy_adjacent_list_iter(g_, iter);
    return degree;
#endif
  }

  inline int GetLocalInDegree(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto al = v2iadj_.at(internal_id);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return static_cast<int>(grin_get_adjacent_list_size(g_, al));
#else
    int degree = 0;
    auto iter = grin_get_adjacent_list_begin(g_, al);
    while (!grin_is_adjacent_list_end(g_, iter)) {
      ++degree;
      grin_get_next_adjacent_list_iter(g_, iter);
    }
    grin_destroy_adjacent_list_iter(g_, iter);
    return degree;
#endif
  }

  inline grape::DestList IEDests(const vertex_t& v) const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto pos = grin_get_position_of_vertex_from_sorted_list(g_, ivl_, v.grin_v);
    return grape::DestList(idoffset_[pos],
                           idoffset_[pos + 1]);
#else
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto pos = iv2i_->at(internal_id);
    return grape::DestList(idoffset_[pos],
                           idoffset_[pos + 1]);
#endif
  }

  inline grape::DestList OEDests(const vertex_t& v) const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto pos = grin_get_position_of_vertex_from_sorted_list(g_, ivl_, v.grin_v);
    return grape::DestList(odoffset_[pos],
                           odoffset_[pos + 1]);
#else
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto pos = iv2i_->at(internal_id);
    return grape::DestList(odoffset_[pos],
                           odoffset_[pos + 1]);
#endif
  }

  inline grape::DestList IOEDests(const vertex_t& v) const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto pos = grin_get_position_of_vertex_from_sorted_list(g_, ivl_, v.grin_v);
    return grape::DestList(iodoffset_[pos],
                           iodoffset_[pos + 1]);
#else
    auto internal_id = grin_get_vertex_internal_id(g_, v.grin_v);
    auto pos = iv2i_->at(internal_id);
    return grape::DestList(iodoffset_[pos],
                           iodoffset_[pos + 1]);
#endif
  }

  void initDestFidList(
      bool in_edge, bool out_edge,
      std::vector<fid_t>& fid_list,
      std::vector<fid_t*>& fid_list_offset) {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
      if (!fid_list_offset.empty()) {
        return;
      }

      fid_list_offset.resize(ivnum_ + 1, NULL);
      std::vector<int> id_num(ivnum_, 0);
      std::set<fid_t> dstset;
      for (size_t i = 0; i < ivnum_; ++i) {
        dstset.clear();
        auto v = grin_get_vertex_from_list(g_, ivl_, i);
        auto internal_id = grin_get_vertex_internal_id(g_, v);
        if (in_edge) {
          auto al = v2iadj_.at(internal_id);
          auto sz = grin_get_adjacent_list_size(g_, al);
          for (size_t j = 0; j < sz; ++j) {
            auto neighbor = grin_get_neighbor_from_adjacent_list(g_, al, j);
            auto v_ref = grin_get_vertex_ref_by_vertex(g_, neighbor);
            auto p = grin_get_master_partition_from_vertex_ref(g_, v_ref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
            grin_destroy_partition(g_, p);
            grin_destroy_vertex_ref(g_, v_ref);
            grin_destroy_vertex(g_, neighbor);
          }
        }
        if (out_edge) {
          auto al = v2oadj_.at(internal_id);
          auto sz = grin_get_adjacent_list_size(g_, al);
          for (size_t j = 0; j < sz; ++j) {
            auto neighbor = grin_get_neighbor_from_adjacent_list(g_, al, j);
            auto v_ref = grin_get_vertex_ref_by_vertex(g_, neighbor);
            auto p = grin_get_master_partition_from_vertex_ref(g_, v_ref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
            grin_destroy_partition(g_, p);
            grin_destroy_vertex_ref(g_, v_ref);
            grin_destroy_vertex(g_, neighbor);
          }
        }
        id_num[i] = dstset.size();
        for (auto fid : dstset) {
          fid_list.push_back(fid);
        }
      }

      fid_list.shrink_to_fit();
      fid_list_offset[0] = fid_list.data();
      for (size_t i = 0; i < ivnum_; ++i) {
        fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
      }
#else  // GRIN_ENABLE_VERTEX_LIST_ARRAY
      if (!fid_list_offset.empty()) {
        return;
      }
      fid_list_offset.resize(ivnum_ + 1, NULL);
      std::vector<int> id_num(ivnum_, 0);
      std::set<fid_t> dstset;
      size_t index = 0;
      auto iv_iter = grin_get_vertex_list_begin(g_, ivl_);
      while (!grin_is_vertex_list_end(g_, iv_iter)) {
        dstset.clear();
        auto v = grin_get_vertex_from_iter(g_, iv_iter);
        auto internal_id = grin_get_vertex_internal_id(g_, v);
        if (in_edge) {
          auto al = v2iadj_.at(internal_id);
          auto e_iter = grin_get_adjacent_list_begin(g_, al);
          while (!grin_is_adjacent_list_end(g, e_iter)) {
            auto neighbor = grin_get_neighbor_from_adjacent_list_iter(g, ali);
            auto v_ref = grin_get_vertex_ref_by_vertex(g_, neighbor);
            auto p = grin_get_master_partition_from_vertex_ref(g_, v_ref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
            grin_destroy_partition(g_, p);
            grin_destroy_vertex_ref(g_, v_ref);
            grin_destroy_vertex(g_, neighbor);
            grin_get_next_adjacent_list_iter(g, e_iter);
          }
        }
        if (out_edge) {
          auto al = v2oadj_.at(internal_id);
          auto e_iter = grin_get_adjacent_list_begin(g_, al);
          while (!grin_is_adjacent_list_end(g, e_iter)) {
            auto neighbor = grin_get_neighbor_from_adjacent_list_iter(g, ali);
            auto v_ref = grin_get_vertex_ref_by_vertex(g_, neighbor);
            auto p = grin_get_master_partition_from_vertex_ref(g_, v_ref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
            grin_destroy_partition(g_, p);
            grin_destroy_vertex_ref(g_, v_ref);
            grin_destroy_vertex(g_, neighbor);
            grin_get_next_adjacent_list_iter(g, e_iter);
          }
        }
        id_num[index] = dstset.size();
        for (auto fid : dstset) {
          fid_list.push_back(fid);
        }
        ++index;
        grin_get_next_vertex_list_iter(g_, v_iter);
      }

      fid_list.shrink_to_fit();
      fid_list_offset[0] = fid_list.data();
      for (size_t i = 0; i < ivnum_; ++i) {
        fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
      }
#endif
    }

  void PrepareToRunApp(const grape::CommSpec& comm_spec, grape::PrepareConf conf) {
    if (conf.message_strategy ==
        grape::MessageStrategy::kAlongEdgeToOuterVertex) {
      initDestFidList(true, true, iodst_, iodoffset_);
    } else if (conf.message_strategy ==
              grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      initDestFidList(true, false, idst_, idoffset_);
    } else if (conf.message_strategy ==
              grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initDestFidList(false, true, odst_, odoffset_);
    }
  }

 private:
  GRIN_PARTITIONED_GRAPH pg_;
  GRIN_GRAPH g_;
  GRIN_PARTITION partition_;
  std::string v_prop_, e_prop_;
  GRIN_VERTEX_PROPERTY vp_;
  GRIN_EDGE_PROPERTY ep_;
  fid_t fid_;
  fid_t fnum_;

  std::vector<fid_t> idst_, odst_, iodst_;
  std::vector<fid_t*> idoffset_, odoffset_,
      iodoffset_;
  GRIN_EDGE_TYPE_LIST etl_;
  std::unordered_map<internal_id_t, GRIN_ADJACENT_LIST> v2iadj_;
  std::unordered_map<internal_id_t, GRIN_ADJACENT_LIST> v2oadj_;

  GRIN_VERTEX_LIST ivl_, ovl_, tvl_;
  GRIN_VERTEX_TYPE_LIST vtl_;

  // std::shared_ptr<std::vector<GRIN_EDGE_PROPERTY_TABLE>> epts_;

  size_t ivnum_;
  size_t ovnum_;
  size_t tvnum_;
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_H_
