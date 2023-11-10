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

#ifndef ANALYTICAL_ENGINE_CORE_GRIN_GRIN_PROJECTED_FRAGMENT_H_
#define ANALYTICAL_ENGINE_CORE_GRIN_GRIN_PROJECTED_FRAGMENT_H_

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "grape/fragment/fragment_base.h"
#include "grape/types.h"
#include "vineyard/common/util/config.h"

#include "core/grin/fragment/grin_util.h" 
#include "core/config.h"
#include "proto/types.pb.h"

namespace grape {
class CommSpec;
}

namespace gs {

/**
 * @brief This class represents the fragment projected from ArrowFragment which
 * contains only one vertex label and edge label. The fragment has no label and
 * property.
 *
 * @tparam OID_T OID type
 * @tparam VID_T VID type
 * @tparam VDATA_T The type of data attached with the vertex
 * @tparam EDATA_T The type of data attached with the edge
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class GRINProjectedFragment {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using gid_t = vid_t;
  using fid_t = GRIN_PARTITION_ID;
  using internal_id_t = int64_t;
  using vertex_range_t = grin_util::VertexRange;
  using inner_vertices_t = vertex_range_t;
  using outer_vertices_t = vertex_range_t;
  using vertices_t = vertex_range_t;
  using sub_vertices_t = vertex_range_t;

  using vertex_t = grin_util::Vertex;
  using adj_list_t =
      grin_util::AdjList<EDATA_T>;
  using const_adj_list_t = adj_list_t;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;

  template <typename DATA_T>
  using vertex_array_t = grin_util::VertexArray<DATA_T>;

  template <typename DATA_T>
  using inner_vertex_array_t = grin_util::VertexArray<DATA_T>;

  template <typename DATA_T>
  using outer_vertex_array_t = grin_util::VertexArray<DATA_T>;

  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  ~GRINProjectedFragment() {
    grin_destroy_vertex_list(g_, tvl_);
    grin_destroy_vertex_list(g_, ivl_);
    grin_destroy_vertex_list(g_, ovl_);
    for (const auto& pair : v2iadj_) {
      grin_destroy_adjacent_list(g_, pair.second);
    }
    for (const auto& pair : v2oadj_) {
      grin_destroy_adjacent_list(g_, pair.second);
    }
    /*
    grin_destropy_vertex_type(g_, vt_);
    grin_destropy_edge_type(g_, et_);
    grin_destropy_vertex_property(g_, vp_);
    grin_destropy_edge_property(g_, ep_);
    */
    grin_destroy_graph(g_);
  }

#if defined(GRIN_WITH_VERTEX_PROPERTY) && defined(GRIN_WITH_EDGE_PROPERTY)
  explicit GRINProjectedFragment(
      GRIN_PARTITIONED_GRAPH partitioned_graph, GRIN_PARTITION partition,
      const std::string& v_label, const std::string& v_prop,
      const std::string& e_label, const std::string& e_prop)
      : pg_(partitioned_graph), partition_(partition) {
    fid_ = grin_get_partition_id(pg_, partition_);
    fnum_ = grin_get_total_partitions_number(pg_);
    g_ = grin_get_local_graph_by_partition(partitioned_graph, partition);
    tvnum_ = ivnum_ = ovnum_ = 0;
    vt_ = grin_get_vertex_type_by_name(g_, v_label.c_str());
    et_ = grin_get_edge_type_by_name(g_, e_label.c_str());
    tvl_ = grin_get_vertex_list_by_type(g_, vt_);
    ivl_ = grin_get_vertex_list_by_type_select_master(g_, vt_);
    ovl_ = grin_get_vertex_list_by_type_select_mirror(g_, vt_);
    vp_ = grin_get_vertex_property_by_name(g_, vt_, v_prop.c_str());
    if (vp_ == GRIN_NULL_VERTEX_PROPERTY) {
      LOG(FATAL) << "Vertex property " << v_prop << " not exist.";
    }
    ep_ = grin_get_edge_property_by_name(g_, et_, e_prop.c_str());
    if (ep_ == GRIN_NULL_EDGE_PROPERTY) {
      LOG(FATAL) << "Edge property " << e_prop << " not exist.";
    }
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    tvnum_ = grin_get_vertex_list_size(g_, tvl_);
    ivnum_ = grin_get_vertex_list_size(g_, ivl_);
    ovnum_ = grin_get_vertex_list_size(g_, ovl_);

    for (size_t i = 0; i < ivnum_; ++i) {
      auto v = grin_get_vertex_from_list(g_, ivl_, i);
      auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v);
      v2iadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v, et_);
      v2oadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v, et_);
      grin_destroy_vertex(g_, v);
    }
#elif defined(GRIN_ENABLE_VERTEX_LIST_ITERATOR)
    auto iv_iter = grin_get_vertex_list_begin(g_, ivl_);
    while (!grin_is_vertex_list_end(g_, iv_iter)) {
        auto v = grin_get_vertex_from_iter(g_, iv_iter);
        auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v);
        ++ivnum_;
        v2iadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v, et_);
        v2oadj_[internal_id] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v, et_);
        grin_destroy_vertex(g_, v);
        grin_get_next_vertex_list_iter(g_, iv_iter);
    }
    grin_destroy_vertex_list_iter(g_, iv_iter);
    auto ov_iter = grin_get_vertex_list_begin(g_, ovl_);
    while (!grin_is_vertex_list_end(g_, ov_iter)) {
      ++ovnum_;
      grin_get_next_vertex_list_iter(g_, ov_iter);
    }
    grin_destroy_vertex_list_iter(g_, ov_iter);
    tvnum_ = ivnum_ + ovnum_;
#endif
  }
#endif

  void PrepareToRunApp(const grape::CommSpec& comm_spec,
                       grape::PrepareConf conf) {
    if (conf.message_strategy ==
        grape::MessageStrategy::kAlongEdgeToOuterVertex) {
      initDestFidListSeq(true, true, iodst_, iodoffset_);
    } else if (conf.message_strategy ==
               grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      initDestFidListSeq(true, false, idst_, idoffset_);
    } else if (conf.message_strategy ==
               grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initDestFidListSeq(false, true, odst_, odoffset_);
    }

    // initOuterVertexRanges();
  }

  inline fid_t fid() const { return fid_; }

  inline fid_t fnum() const { return fnum_; }

  inline vertex_range_t Vertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, tvl_, vt_, 0, tvnum_);
#else
#endif
  }

  inline vertex_range_t InnerVertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, ivl_, vt_, 0, ivnum_);
#else
#endif
  }

  inline vertex_range_t OuterVertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, ovl_, vt_, 0, ovnum_);
#else
#endif
  }

#ifdef GRIN_ENABLE_VERTEX_EXTERNAL_ID_OF_INT64
  inline bool GetVertex(const oid_t& oid, vertex_t& v) const {
    auto grin_v = grin_get_vertex_by_external_id_of_int64(g_, oid);
    if (grin_v == GRIN_NULL_VERTEX) {
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  inline oid_t GetId(const vertex_t& v) const {
    return grin_get_vertex_external_id_of_int64(g_, v.grin_v);
  }
#endif // GRIN_ENABLE_VERTEX_EXTERNAL_ID_OF_INT64

  inline GRIN_PARTITION_ID GetFragId(const vertex_t& v) const {
    auto v_ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    auto partition = grin_get_master_partition_from_vertex_ref(g_, v_ref);
    auto fid = grin_get_partition_id(pg_, partition);
    grin_destroy_vertex_ref(g_, v_ref);
    grin_destroy_partition(g_, partition);
    return fid;
  }

  vdata_t GetData(const vertex_t& v) const {
    return grin_get_vertex_property_value_of_int64(g_, v.grin_v, vp_);
  }

  inline bool Gid2Vertex(const gid_t& gid, vertex_t& v) const {
    auto v_ref = grin_deserialize_int64_to_vertex_ref(g_, gid);
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

  inline gid_t Vertex2Gid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline size_t GetInnerVerticesNum() const { return ivnum_; }

  inline size_t GetOuterVerticesNum() const { return ovnum_; }

  inline size_t GetVerticesNum() const { return tvnum_; }

  inline size_t GetEdgeNum() const { return ienum_ + oenum_; }

  inline size_t GetInEdgeNum() const { return ienum_; }

  inline size_t GetOutEdgeNum() const { return oenum_; }

  /* Get outgoing edges num from this frag*/
  inline size_t GetOutgoingEdgeNum() const {
    return oenum_;
  }

  /* Get incoming edges num to this frag*/
  inline size_t GetIncomingEdgeNum() const {
    return ienum_;
  }

  inline bool IsInnerVertex(const vertex_t& v) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    return grin_is_master_vertex(g_, v.grin_v);
  }

  inline bool IsOuterVertex(const vertex_t& v) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    return grin_is_mirror_vertex(g_, v.grin_v);
  }

  inline bool GetInnerVertex(const oid_t& oid, vertex_t& v) const {
    // TODO: may be just get vertex by label index
    auto grin_v = grin_get_vertex_by_external_id_of_int64(g_, oid);
    if (grin_v == GRIN_NULL_VERTEX || !grin_is_master_vertex(g_, grin_v)) {
      grin_destroy_vertex(g_, grin_v);
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  inline bool GetOuterVertex(const oid_t& oid, vertex_t& v) const {
    auto grin_v = grin_get_vertex_by_external_id_of_int64(g_, oid);
     if (grin_v == GRIN_NULL_VERTEX) {
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  inline oid_t GetInnerVertexId(const vertex_t& v) const {
    return grin_get_vertex_external_id_of_int64(g_, v.grin_v);
  }

  inline oid_t GetOuterVertexId(const vertex_t& v) const {
    return grin_get_vertex_external_id_of_int64(g_, v.grin_v);
  }

/*
  inline oid_t Gid2Oid(const vid_t& gid) const {
    internal_oid_t internal_oid;
    CHECK(vm_ptr_->GetOid(gid, internal_oid));
    return oid_t(internal_oid);
  }

  inline bool Oid2Gid(const oid_t& oid, vid_t& gid) const {
    return vm_ptr_->GetGid(internal_oid_t(oid), gid);
  }

  // For Java use, can not use Oid2Gid(const oid_t & oid, vid_t & gid) since
  // Java can not pass vid_t by reference.
  inline vid_t Oid2Gid(const oid_t& oid) const {
    vid_t gid;
    if (vm_ptr_->GetGid(internal_oid_t(oid), gid)) {
      return gid;
    }
    return std::numeric_limits<vid_t>::max();
  }
  */

  inline bool InnerVertexGid2Vertex(const gid_t& gid, vertex_t& v) const {
    auto v_ref = grin_deserialize_int64_to_vertex_ref(g_, gid);
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

  inline bool OuterVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    auto v_ref = grin_deserialize_int64_to_vertex_ref(g_, gid);
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

  inline gid_t GetOuterVertexGid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline gid_t GetInnerVertexGid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    auto al = v2iadj_.at(internal_id);
    if (v2iadj_.find(internal_id) == v2iadj_.end()) {
      LOG(FATAL) << "Get in faild: " << internal_id;
    }
    // auto al = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v.grin_v, et_);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto sz = grin_get_adjacent_list_size(g_, al);
    return adj_list_t(g_, al, ep_, 0, sz);
#else
#endif
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    auto al = v2oadj_.at(internal_id);
    // auto al = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v.grin_v, et_);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto sz = grin_get_adjacent_list_size(g_, al);
    return adj_list_t(g_, al, ep_, 0, sz);
#else
#endif
  }

  inline adj_list_t WrapGetOutgoingAdjList(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    auto al = v2oadj_.at(internal_id);
    // auto al = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v.grin_v, et_);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto sz = grin_get_adjacent_list_size(g_, al);
    return adj_list_t(g_, al, ep_, 0, sz);
#else
#endif
  }

  inline adj_list_t WrapGetIncomingAdjList(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    auto al = v2iadj_.at(internal_id);
    if (v2iadj_.find(internal_id) == v2iadj_.end()) {
      LOG(FATAL) << "Get in faild: " << internal_id;
    }
    // auto al = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v.grin_v, et_);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto sz = grin_get_adjacent_list_size(g_, al);
    return adj_list_t(g_, al, ep_, 0, sz);
#else
#endif
  }

  inline int GetLocalOutDegree(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
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
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
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
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    return grape::DestList(idoffset_[internal_id],
                           idoffset_[internal_id + 1]);
  }

  inline grape::DestList OEDests(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    return grape::DestList(odoffset_[internal_id],
                           odoffset_[internal_id + 1]);
  }

  inline grape::DestList IOEDests(const vertex_t& v) const {
    auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
    return grape::DestList(iodoffset_[internal_id],
                           iodoffset_[internal_id + 1]);
  }

  inline bool directed() const { return directed_; }

 private:
  void initDestFidList(const grape::CommSpec& comm_spec, const bool in_edge,
                       const bool out_edge, std::vector<fid_t>& fid_list,
                       std::vector<fid_t*>& fid_list_offset) {
    /*
    if (!fid_list_offset.empty()) {
      return;
    }
    fid_list_offset.resize(ivnum_ + 1, NULL);

    int concurrency =
        (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
        comm_spec.local_num();

    // don't use std::vector<bool> due to its specialization
    std::vector<uint8_t> fid_list_bitmap(ivnum_ * fnum_, 0);
    std::atomic_size_t fid_list_size(0);

    vineyard::parallel_for(
        static_cast<vid_t>(0), static_cast<vid_t>(ivnum_),
        [this, in_edge, out_edge, &fid_list_bitmap,
         &fid_list_size](const vid_t& offset) {
          vertex_t v = *(inner_vertices_.begin() + offset);
          if (in_edge) {
            auto es = GetIncomingAdjList(v);
            fid_t last_fid = -1;
            for (auto& e : es) {
              fid_t f = GetFragId(e.neighbor());
              if (f != last_fid && f != fid_ &&
                  !fid_list_bitmap[offset * fnum_ + f]) {
                last_fid = f;
                fid_list_bitmap[offset * fnum_ + f] = 1;
                fid_list_size.fetch_add(1);
              }
            }
          }
          if (out_edge) {
            auto es = GetOutgoingAdjList(v);
            fid_t last_fid = -1;
            for (auto& e : es) {
              fid_t f = GetFragId(e.neighbor());
              if (f != last_fid && f != fid_ &&
                  !fid_list_bitmap[offset * fnum_ + f]) {
                last_fid = f;
                fid_list_bitmap[offset * fnum_ + f] = 1;
                fid_list_size.fetch_add(1);
              }
            }
          }
        },
        concurrency, 1024);

    fid_list.reserve(fid_list_size.load());
    fid_list_offset[0] = fid_list.data();

    for (vid_t i = 0; i < ivnum_; ++i) {
      size_t nonzero = 0;
      for (fid_t fid = 0; fid < fnum_; ++fid) {
        if (fid_list_bitmap[i * fnum_ + fid]) {
          nonzero += 1;
          fid_list.push_back(fid);
        }
      }
      fid_list_offset[i + 1] = fid_list_offset[i] + nonzero;
    }
    */
  }

  void initDestFidListSeq(const bool in_edge, const bool out_edge,
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
        auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v);
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
        auto internal_id = grin_get_vertex_internal_id_by_type(g_, vt_, v);
        if (in_edge) {
          auto al = v2iadj_.at(internal_id);
          auto e_iter = grin_get_adjacent_list_begin(g_, al);
          while (!grin_is_adjacent_list_end(g_, e_iter)) {
            auto neighbor = grin_get_neighbor_from_adjacent_list_iter(g_, e_iter);
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
            grin_get_next_adjacent_list_iter(g_, e_iter);
          }
        }
        if (out_edge) {
          auto al = v2oadj_.at(internal_id);
          auto e_iter = grin_get_adjacent_list_begin(g_, al);
          while (!grin_is_adjacent_list_end(g_, e_iter)) {
            auto neighbor = grin_get_neighbor_from_adjacent_list_iter(g_, e_iter);
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
            grin_get_next_adjacent_list_iter(g_, e_iter);
          }
        }
        id_num[index] = dstset.size();
        for (auto fid : dstset) {
          fid_list.push_back(fid);
        }
        ++index;
        grin_get_next_vertex_list_iter(g_, iv_iter);
      }

      fid_list.shrink_to_fit();
      fid_list_offset[0] = fid_list.data();
      for (size_t i = 0; i < ivnum_; ++i) {
        fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
      }
#endif
  }

 private:
  GRIN_PARTITIONED_GRAPH pg_;
  GRIN_GRAPH g_;
  GRIN_PARTITION partition_;
  fid_t fid_, fnum_;
  GRIN_VERTEX_PROPERTY vp_;
  GRIN_EDGE_PROPERTY ep_;
  bool directed_;

  vid_t ivnum_, ovnum_, tvnum_;
  size_t ienum_{}, oenum_{};

  std::vector<fid_t> idst_, odst_, iodst_;
  std::vector<fid_t*> idoffset_, odoffset_,
      iodoffset_;

  GRIN_VERTEX_TYPE vt_;
  GRIN_EDGE_TYPE et_;
  std::unordered_map<internal_id_t, GRIN_ADJACENT_LIST> v2iadj_;
  std::unordered_map<internal_id_t, GRIN_ADJACENT_LIST> v2oadj_;

  GRIN_VERTEX_LIST ivl_, ovl_, tvl_;
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_PROJECTED_FRAGMENT_H_
