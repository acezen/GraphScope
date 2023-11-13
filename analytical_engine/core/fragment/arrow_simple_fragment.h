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

#ifndef ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_H_
#define ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_H_

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/types.h"
#include "grape/utils/vertex_array.h"
#include "vineyard/graph/fragment/arrow_fragment.h"

#include "core/config.h"

namespace grape {
class CommSpec;
}

namespace gs {

namespace arrow_simple_fragment_impl {


template <typename T>
class VertexRange {
 public:
  using vertex_t = grape::Vertex<T>;

  VertexRange() {}
  VertexRange(const T& begin, size_t start, size_t end) 
      : begin_(begin), start_(start), end_(end) {}
  VertexRange(const VertexRange& r) : begin_(r.begin_), start_(r.start_), end_(r.end_) {}

  class iterator {
    using reference_type = grape::Vertex<T>;

   private:
    T begin_;
    size_t cur_;

   public:
    iterator() noexcept : cur_() {}
    explicit iterator(const T& begin, const T& cur) noexcept : begin_(begin), cur_(cur) {}

    inline reference_type operator*() noexcept { return grape::Vertex(begin_ + cur_); }

    inline iterator& operator++() noexcept {
      ++cur_;
      return *this;
    }

    inline iterator operator++(int) noexcept {
      iterator ret = *this;
      ++*this;
      return ret;
    }

    inline iterator& operator--() noexcept {
      --cur_;
      return *this;
    }

    inline iterator operator--(int) noexcept {
      iterator ret = *this;
      --*this;
      return ret;
    }

    inline iterator operator+(size_t offset) const noexcept {
      return iterator(begin_, cur_ + offset);
    }

    inline bool operator==(const iterator& rhs) const noexcept {
      return cur_ == rhs.cur_;
    }

    inline bool operator!=(const iterator& rhs) const noexcept {
      return cur_ != rhs.cur_;
    }

    inline bool operator<(const iterator& rhs) const noexcept {
      return cur_ < rhs.cur_;
    }
  };

  inline iterator begin() const { return iterator(begin_, start_); }

  inline iterator end() const { return iterator(begin_, end_); }

  inline size_t size() const { return end_ - start_; }

  void Swap(VertexRange& rhs) {
#ifdef __CUDACC__
    thrust::swap(begin_, rhs.begin_);
    thrust::swap(end_, rhs.end_);
#else
    std::swap(begin_, rhs.begin_);
    std::swap(end_, rhs.end_);
#endif
  }

  void SetRange(size_t start, size_t end) {
    start_ = start_; 
    end_ = end;
  }

  const T& begin_value() const { return begin_; }

  const T& end_value() const { return end_; }

  inline bool Contain(const grape::Vertex<T>& v) const {
    return begin_ <= v.GetValue() && v.GetValue() < end_;
  }

 private:
  T begin_;
  size_t start_, end_;
};

template <typename T>
bool operator<(grape::Vertex<T> const& lhs, grape::Vertex<T> const& rhs) {
  return lhs.GetValue() < rhs.GetValue();
}

template <typename T>
bool operator==(grape::Vertex<T> const& lhs, grape::Vertex<T> const& rhs) {
  return lhs.GetValue() == rhs.GetValue();
}

template <typename VERTEX_SET_T, typename T>
class VertexArray {};

template <typename VID_T, typename T>
class VertexArray<VertexRange<VID_T>, T> : public grape::Array<T, grape::Allocator<T>> {
  using Base = grape::Array<T, grape::Allocator<T>>;

 public:
  VertexArray() : Base(), fake_start_(NULL) {
    id_parser_.Init(1, 1);
  }
  explicit VertexArray(const VertexRange<VID_T>& range)
      : Base(range.size()), range_(range) {
    fake_start_ = Base::data();
    id_parser_.Init(1, 1);
  }
  VertexArray(const VertexRange<VID_T>& range, const T& value)
      : Base(range.size(), value), range_(range) {
    fake_start_ = Base::data();
    id_parser_.Init(1, 1);
  }

  ~VertexArray() = default;

  void Init(const VertexRange<VID_T>& range) {
    Base::clear();
    Base::resize(range.size());
    range_ = range;
    fake_start_ = Base::data();
  }

  void Init(const VertexRange<VID_T>& range, const T& value) {
    Base::clear();
    Base::resize(range.size(), value);
    range_ = range;
    fake_start_ = Base::data();
  }

  void SetValue(VertexRange<VID_T>& range, const T& value) {
    std::fill_n(&Base::data()[range.begin_value() - range_.begin_value()],
                range.size(), value);
  }
  void SetValue(const grape::Vertex<VID_T>& loc, const T& value) {
    auto internal_id = id_parser_.GetOffset(loc.GetValue());
    fake_start_[internal_id] = value;
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](const grape::Vertex<VID_T>& loc) {
    auto internal_id = id_parser_.GetOffset(loc.GetValue());
    return fake_start_[internal_id];
  }
  inline const T& operator[](const grape::Vertex<VID_T>& loc) const {
    auto internal_id = id_parser_.GetOffset(loc.GetValue());
    return fake_start_[internal_id];
  }

  void Swap(VertexArray& rhs) {
    Base::swap((Base&) rhs);
    range_.Swap(rhs.range_);
    std::swap(fake_start_, rhs.fake_start_);
  }

  void Clear() {
    VertexArray ga;
    this->Swap(ga);
  }

  const VertexRange<VID_T>& GetVertexRange() const { return range_; }

 private:
  void Resize() {}

  VertexRange<VID_T> range_;
  T* fake_start_;
  vineyard::IdParser<VID_T> id_parser_;
};

template <typename VID_T, typename EID_T, typename EDATA_T>
class NbrDefault {
  using nbr_t = vineyard::property_graph_utils::Nbr<VID_T, EID_T>;
  using prop_id_t = vineyard::property_graph_types::PROP_ID_TYPE;

 public:
  explicit NbrDefault(const prop_id_t& default_prop_id)
      : default_prop_id_(default_prop_id) {}
  NbrDefault(const nbr_t& begin, size_t cur, const prop_id_t& default_prop_id)
      : begin_(begin),
        cur_(cur),
        default_prop_id_(default_prop_id) {}
  NbrDefault(const NbrDefault& rhs)
      : begin_(rhs.begin_),
        cur_(rhs.cur_),
        default_prop_id_(rhs.default_prop_id_) {}
  NbrDefault(NbrDefault&& rhs)
      : begin_(rhs.begin_),
        cur_(rhs.cur_),
        default_prop_id_(rhs.default_prop_id_) {}

  inline NbrDefault& operator=(const NbrDefault& rhs) {
    begin_ = rhs.begin_;
    cur_ = rhs.cur_;
    default_prop_id_ = rhs.default_prop_id_;
    return *this;
  }

  inline NbrDefault& operator=(NbrDefault&& rhs) {
    begin_ = std::move(rhs.begin_);
    cur_ = rhs.cur_;
    default_prop_id_ = rhs.default_prop_id_;
    return *this;
  }

  grape::Vertex<VID_T> neighbor() const {
    auto nbr = begin_ + cur_;
    return grape::Vertex<VID_T>(nbr.neighbor().GetValue());
  }

  grape::Vertex<VID_T> get_neighbor() const {
    auto nbr = begin_ + cur_;
    return grape::Vertex<VID_T>(nbr.neighbor().GetValue());
  }

  EDATA_T get_data() const {
    auto nbr = begin_ + cur_;
    return nbr.template get_data<EDATA_T>(default_prop_id_);
  }

  inline const NbrDefault& operator++() {
    ++cur_;
    return *this;
  }

  inline NbrDefault operator++(int) {
    NbrDefault ret(begin_, cur_, default_prop_id_);
    ++(*this);
    return ret;
  }

  inline NbrDefault& operator--() {
    --cur_;
    return *this;
  }

  inline NbrDefault operator--(int) {
    NbrDefault ret(begin_, cur_, default_prop_id_);
    --(*this);
    return ret;
  }

  inline bool operator==(const NbrDefault& rhs) const {
    return begin_ == rhs.begin_ && cur_ == rhs.cur_;
  }
  inline bool operator!=(const NbrDefault& rhs) const {
    return begin_ != rhs.begin_ || cur_ != rhs.cur_;
  }
  inline bool operator<(const NbrDefault& rhs) const { return begin_ == rhs.begin_ && cur_ < rhs.cur_; }

  inline const NbrDefault& operator*() const { return *this; }

 private:
  nbr_t begin_;
  size_t cur_;
  prop_id_t default_prop_id_;
};

template <typename VID_T, typename EID_T, typename EDATA_T>
class WrapAdjList {
  using vid_t = VID_T;
  using eid_t = EID_T;
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<vid_t, eid_t>;
  using adj_list_t = vineyard::property_graph_utils::AdjList<VID_T, EID_T>;
  using prop_id_t = vineyard::property_graph_types::PROP_ID_TYPE;

 public:
  WrapAdjList() {}

  explicit WrapAdjList(const adj_list_t& adj_list,
                       size_t begin, size_t end, const prop_id_t& prop_id)
      : adj_list_(adj_list),
        begin_(begin), end_(end), prop_id_(prop_id),{}

  NbrDefault<VID_T, EID_T, EDATA_T> begin() const {
    return NbrDefault<VID_T, EID_T, EDATA_T>(adj_list_.begin(), begin_, prop_id_);
  }

  NbrDefault<VID_T, EID_T, EDATA_T> end() const {
    return NbrDefault<VID_T, EID_T, EDATA_T>(adj_list_.begin(), end_, prop_id_);
  }

  inline size_t Size() const { return adj_list_.Size(); }

  inline bool Empty() const { return adj_list_.Empty(); }

  inline bool NotEmpty() const { return adj_list_.NotEmpty(); }

 private:
  adj_list_t adj_list_;
  size_t begin_;
  size_t end_;
  prop_id_t prop_id_;
};

}  // namespace arrow_flattened_fragment_impl

/**
 * @brief This class represents the fragment flattened from ArrowFragment.
 * Different from ArrowProjectedFragment, an ArrowSimpleFragment derives from
 * an ArrowFragment, but flattens all the labels to one type, result in a graph
 * with a single type of vertices and a single type of edges. Optionally,
 * a common property across labels of vertices(reps., edges) in the
 * ArrowFragment will be reserved as vdata(resp, edata).
 * ArrowSimpleFragment usually used as a wrapper for ArrowFragment to run the
 * applications/algorithms defined in NetworkX or Analytical engine,
 * since these algorithms need the topology of the whole (property) graph.
 *
 * @tparam OID_T
 * @tparam VID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T = vineyard::ArrowVertexMap<
              typename vineyard::InternalType<OID_T>::type, VID_T>>
class ArrowSimpleFragment {
 public:
  // TODO(tao): ArrowFragment with compact edges cannot be flattened.
  using fragment_t = vineyard::ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, false>;
  using simple_fragment_t =
      ArrowSimpleFragment<OID_T, VID_T, VDATA_T, EDATA_T, VERTEX_MAP_T>;
  using oid_t = OID_T;
  using vid_t = VID_T;
  using internal_oid_t = typename vineyard::InternalType<oid_t>::type;
  using eid_t = typename fragment_t::eid_t;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = typename fragment_t::vertex_t;
  using fid_t = grape::fid_t;
  using label_id_t = typename fragment_t::label_id_t;
  using prop_id_t = vineyard::property_graph_types::PROP_ID_TYPE;
  using vertex_range_t = arrow_simple_fragment_impl::VertexRange<vid_t>;
  using inner_vertices_t = vertex_range_t;
  using outer_vertices_t = vertex_range_t;
  using vertices_t = vertex_range_t;

  template <typename DATA_T>
  using vertex_array_t = arrow_simple_fragment_impl::VertexArray<vertices_t, DATA_T>;

  template <typename DATA_T>
  using inner_vertex_array_t = arrow_simple_fragment_impl::VertexArray<inner_vertices_t, DATA_T>;

  template <typename DATA_T>
  using outer_vertex_array_t = arrow_simple_fragment_impl::VertexArray<outer_vertices_t, DATA_T>;

  using adj_list_t =
      arrow_simple_fragment_impl::WrapAdjList<vid_t, eid_t, edata_t>;
  
  using dest_list_t = grape::DestList;

  // This member is used by grape::check_load_strategy_compatible()
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  ArrowSimpleFragment() = default;

  explicit ArrowSimpleFragment(std::shared_ptr<fragment_t> frag, const label_id_t& v_label, const label_id_t& e_label, 
      const prop_id_t& v_prop, const prop_id_t& e_prop)
      : fragment_(frag),
        vertex_label_(v_label),
        edge_label_(e_label),
        vertex_prop_(e_prop),
        edge_prop_(v_prop) {
    ivnum_ = ovnum_ = tvnum_ = 0;
    ivnum_ = fragment_->GetInnerVerticesNum(v_label);
    ovnum_ = fragment_->GetOuterVerticesNum(v_label);
    tvnum_ = fragment_->GetVerticesNum(v_label);
  }

  virtual ~ArrowSimpleFragment() = default;

  static std::shared_ptr<
      ArrowSimpleFragment<OID_T, VID_T, VDATA_T, EDATA_T, VERTEX_MAP_T>>
  Make(const std::shared_ptr<fragment_t>& frag, const label_id_t& v_label, const label_id_t& e_label, 
      const prop_id_t& v_prop, const prop_id_t& e_prop) {
    return std::make_shared<ArrowSimpleFragment>(frag, v_label, e_label, v_prop,
                                                    e_prop);
  }

  void PrepareToRunApp(const grape::CommSpec& comm_spec,
                       grape::PrepareConf conf) {
    fragment_->PrepareToRunApp(comm_spec, conf);
  }

  inline fid_t fid() const { return fragment_->fid(); }

  inline fid_t fnum() const { return fragment_->fnum(); }

  inline bool directed() const { return fragment_->directed(); }

  inline vertex_range_t Vertices() const { 
    auto vr = fragment_->Vertices(vertex_label_);
    return vertex_range_t(vr.begin_value(), 0, vr.size());
  }

  inline vertex_range_t InnerVertices() const {
    auto vr = fragment_->InnerVertices(vertex_label_);
    return vertex_range_t(vr.begin_value(), 0, vr.size());
  }

  inline vertex_range_t OuterVertices() const {
    auto vr = fragment_->OuterVertices(vertex_label_);
    return vertex_range_t(vr.begin_value(), 0, vr.size());
  }

  inline bool GetVertex(const oid_t& oid, vertex_t& v) const {
    return fragment_->GetVertex(vertex_label_, oid, v);
  }

  inline oid_t GetId(const vertex_t& v) const {
    return fragment_->GetId(v);
  }

  inline internal_oid_t GetInternalId(const vertex_t& v) const {
    return fragment_->GetInternalId(v);
  }

  inline fid_t GetFragId(const vertex_t& u) const {
    return fragment_->GetFragId(u);
  }

  inline bool Gid2Vertex(const vid_t& gid, vertex_t& v) const {
    return fragment_->Gid2Vertex(gid, v);
  }

  inline vid_t Vertex2Gid(const vertex_t& v) const {
    return fragment_->Vertex2Gid(v);
  }

  inline vdata_t GetData(const vertex_t& v) const {
    return fragment_->template GetData<vdata_t>(v, vertex_prop_);
  }

  inline vid_t GetInnerVerticesNum() const { return ivnum_; }

  inline vid_t GetOuterVerticesNum() const { return ovnum_; }

  inline vid_t GetVerticesNum() const { return tvnum_; }

  inline size_t GetEdgeNum() const { return fragment_->GetEdgeNum(); }

  inline bool IsInnerVertex(const vertex_t& v) const {
    return fragment_->IsInnerVertex(v);
  }

  inline bool IsOuterVertex(const vertex_t& v) const {
    return fragment_->IsOuterVertex(v);
  }

  inline bool GetInnerVertex(const oid_t& oid, vertex_t& v) const {
    return fragment_->GetInnerVertex(vertex_label_, oid, v);
  }

  inline bool GetOuterVertex(const oid_t& oid, vertex_t& v) const {
    return fragment_->GetOuterVertex(vertex_label_, oid, v);
  }

  inline oid_t GetInnerVertexId(const vertex_t& v) const {
    return fragment_->GetInnerVertexId(v);
  }

  inline oid_t GetOuterVertexId(const vertex_t& v) const {
    return fragment_->GetOuterVertexId(v);
  }

  inline oid_t Gid2Oid(const vid_t& gid) const {
    return fragment_->Gid2Oid(gid);
  }

  inline bool Oid2Gid(const oid_t& oid, vid_t& gid) const {
    return fragment_->Oid2Gid(vertex_label_, oid, gid);
  }

  inline bool InnerVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    return fragment_->InnerVertexGid2Vertex(gid, v);
  }

  inline bool OuterVertexGid2Vertex(const vid_t& gid, vertex_t& v) const {
    return fragment_->OuterVertexGid2Vertex(gid, v);
  }

  inline vid_t GetOuterVertexGid(const vertex_t& v) const {
    return fragment_->GetOuterVertexGid(v);
  }

  inline vid_t GetInnerVertexGid(const vertex_t& v) const {
    return fragment_->GetInnerVertexGid(v);
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v) const {
    auto al = fragment_->GetOutgoingAdjList(v, edge_label_);
    return adj_list_t(al, 0, al.Size(), edge_prop_); 
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v) const {
    auto al = fragment_->GetOutgoingAdjList(v, edge_label_);
    return adj_list_t(al, 0, al.Size(), edge_prop_); 
  }

  inline int GetLocalOutDegree(const vertex_t& v) const {
    return fragment_->GetLocalOutDegree(v, edge_label_);
  }

  inline int GetLocalInDegree(const vertex_t& v) const {
    return fragment_->GetLocalInDegree(v, edge_label_);
  }

  inline dest_list_t IEDests(const vertex_t& v) const {
    return fragment_->IEDests(v, edge_label_);
  }

  inline dest_list_t OEDests(const vertex_t& v) const {
    return fragment_->OEDests(v, edge_label_);
  }

  inline dest_list_t IOEDests(const vertex_t& v) const {
    return fragment_->IOEDests(v, edge_label_);
  }

 private:
  std::shared_ptr<fragment_t> fragment_;
  label_id_t vertex_label_;
  label_id_t edge_label_;
  prop_id_t vertex_prop_;
  prop_id_t edge_prop_;

  vid_t ivnum_;
  vid_t ovnum_;
  vid_t tvnum_;
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_H_
