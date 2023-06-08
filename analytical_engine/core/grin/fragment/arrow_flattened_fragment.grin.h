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

#include "core/object/app_entry.h"
extern "C" {
#include "graph/grin/predefine.h"
#include "grin/include/topology/structure.h"
#include "grin/include/topology/vertexlist.h"
#include "grin/include/topology/adjacentlist.h"

#include "grin/include/partition/partition.h"
#include "grin/include/partition/topology.h"
#include "grin/include/property/topology.h"
#include "grin/include/partition/reference.h"

#include "grin/include/property/type.h"
#include "grin/include/property/property.h"
#include "grin/include/property/propertylist.h"
#include "grin/include/property/topology.h"

#include "grin/include/index/order.h"
#include "grin/include/index/original_id.h"
}

namespace grape {
class CommSpec;
}

namespace gs {

namespace arrow_flattened_fragment_grin_impl {

// A wrapper for GRIN_VERTEX
struct Vertex {
  Vertex(uint64_t a) noexcept : g_(GRIN_NULL_GRAPH), grin_v(GRIN_NULL_VERTEX) {}  // compatible with grape
  Vertex() noexcept : g_(GRIN_NULL_GRAPH), grin_v(GRIN_NULL_VERTEX) {}
  explicit Vertex(GRIN_GRAPH g, GRIN_VERTEX v) noexcept : g_(g), grin_v(v) {}
  Vertex& operator=(Vertex&& rhs) noexcept {
    g_ = rhs.g_;
    grin_v = rhs.grin_v;
    rhs.grin_v = GRIN_NULL_VERTEX;
    return *this;
  }
  Vertex(const Vertex& rhs) : g_(rhs.g_), grin_v(GRIN_NULL_VERTEX) {
    grin_v = rhs.grin_v;
    auto v_ref = grin_get_vertex_ref_by_vertex(g_, rhs.grin_v);
    grin_v = grin_get_vertex_from_vertex_ref(g_, v_ref);
    grin_destroy_vertex_ref(g_, v_ref);
  }
  Vertex(Vertex&& rhs) noexcept : g_(rhs.g_), grin_v(rhs.grin_v) {
    rhs.grin_v = GRIN_NULL_VERTEX;
  }
  Vertex& operator=(const Vertex& rhs) {
    g_ = rhs.g_;
    grin_v = GRIN_NULL_VERTEX;
    auto v_ref = grin_get_vertex_ref_by_vertex(g_, rhs.grin_v);
    grin_v = grin_get_vertex_from_vertex_ref(g_, v_ref);
    grin_destroy_vertex_ref(g_, v_ref);
    return *this;
  }
  ~Vertex() {
    grin_destroy_vertex(g_, grin_v);
  }
  bool operator<(const Vertex& rhs) const {
    return grin_v < rhs.grin_v;
  }

  void Refresh(GRIN_GRAPH g, GRIN_VERTEX v) {
    grin_destroy_vertex(g_, grin_v);
    g_ = g;
    grin_v = v;
  }

  GRIN_GRAPH g_;
  GRIN_VERTEX grin_v;
};


#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
class VertexRange {
 public:
  VertexRange() noexcept : g_(GRIN_NULL_GRAPH), vl_(GRIN_NULL_LIST), begin_(0), end_(0) {}
  VertexRange(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, const size_t begin, const size_t end)
      noexcept : g_(g), vl_(vl), begin_(begin), end_(end) {}
  VertexRange(const VertexRange& r) noexcept : g_(r.g_), vl_(r.vl_), begin_(r.begin_), end_(r.end_) {}

  ~VertexRange() = default;

  inline VertexRange& operator=(const VertexRange& r) noexcept {
    g_ = r.g_;
    vl_ = r.vl_;
    begin_ = r.begin_;
    end_ = r.end_;
    return *this;
  }

  class iterator {
    using reference_type = Vertex;

   private:
    GRIN_GRAPH g_;
    GRIN_VERTEX_LIST vl_;
    size_t cur_;

   public:
    iterator() noexcept : g_(GRIN_NULL_GRAPH), vl_(GRIN_NULL_LIST), cur_(0) {}
    explicit iterator(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) noexcept : g_(g), vl_(vl), cur_(idx) {}
    iterator(const iterator& rhs) = default;
    iterator& operator=(const iterator& rhs) = default;
    ~iterator() = default;
    reference_type operator*() noexcept {
      return Vertex(g_, grin_get_vertex_from_list(g_, vl_, cur_));
    }

    iterator& operator++() noexcept {
      ++cur_;
      return *this;
    }

    iterator operator++(int) noexcept {
      return iterator(g_, vl_, cur_ + 1);
    }

    iterator& operator--() noexcept {
      --cur_;
      return *this;
    }

    iterator operator--(int) noexcept {
      return iterator(g_, vl_, cur_--);
    }

    iterator operator+(size_t offset) const noexcept {
      return iterator(g_, vl_, cur_ + offset);
    }

    bool operator==(const iterator& rhs) const noexcept {
      return cur_ == rhs.cur_;
    }

    bool operator!=(const iterator& rhs) const noexcept {
      return cur_ != rhs.cur_;
    }

    bool operator<(const iterator& rhs) const noexcept {
      return vl_ == rhs.vl_ && cur_ < rhs.cur_;
    }
  };

  iterator begin() const { return iterator(g_, vl_, begin_); }

  iterator end() const { return iterator(g_, vl_, end_); }

  size_t size() const { return end_ - begin_; }

  void Swap(VertexRange& rhs) {
    std::swap(vl_, rhs.vl_);
    std::swap(begin_, rhs.begin_);
    std::swap(end_, rhs.end_);
  }

  void SetRange(const size_t begin, const size_t end) {
    begin_ = begin;
    end_ = end;
  }

  size_t begin_value() const { return begin_; }

  size_t end_value() const { return end_; }

  size_t GetVertexLoc(const Vertex& v) const {
    return grin_get_position_of_vertex_from_sorted_list(g_, vl_, v.grin_v);
  }

 public:
  GRIN_GRAPH g_;
  GRIN_VERTEX_LIST vl_;
  size_t begin_;
  size_t end_;
};
#else
class VertexRange {
 public:
  VertexRange() noexcept : g_(GRIN_NULL_GRAPH), vl_(GRIN_NULL_LIST), size_(0), v2idx_(nullptr) {}
  VertexRange(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, const std::unordered_map<GRIN_VERTEX, size_t>* v2idx)
      : g_(g), vl_(vl), size_(v2idx->size()), v2idx_(v2idx) {}

  VertexRange(const VertexRange& r) : g_(r.g_), vl_(r.vl_), size_(r.size_), v2idx_(r.v2idx_) {}

  ~VertexRange() = default;

  inline VertexRange& operator=(const VertexRange& r) {
    g_ = r.g_;
    vl_ = r.vl_;
    size_ = r.size_;
    v2idx_ = r.v2idx_;
    return *this;
  }

  class iterator {
    using reference_type = Vertex;

   private:
    GRIN_GRAPH g_;
    GRIN_VERTEX_LIST_ITERATOR cur_;

   public:
    iterator() noexcept : g_(GRIN_NULL_GRAPH), cur_(GRIN_NULL_LIST_ITERATOR) {}
    explicit iterator(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR iter) noexcept : g_(g), cur_(iter) {}
    iterator(const iterator& rhs) = delete;
    iterator& operator=(const iterator& rhs) = delete;
    iterator& operator=(iterator&& rhs) {
      g_ = rhs.g_;
      cur_ = rhs.cur_;
      rhs.cur_ = GRIN_NULL_LIST_ITERATOR;
      return *this;
    }
    ~iterator() {
      grin_destroy_vertex_list_iter(g_, cur_);
    }

    reference_type operator*() noexcept { return Vertex(g_, grin_get_vertex_from_iter(g_, cur_)); }

    iterator& operator++() noexcept {
      grin_get_next_vertex_list_iter(g_, cur_);
      return *this;
    }

    bool is_end() const noexcept {
      return grin_is_vertex_list_end(g_, cur_);
    }
  };

  iterator begin() const { return iterator(g_, grin_get_vertex_list_begin(g_, vl_)); }

  bool IsEnd(const iterator& iter) const { return iter.is_end(); }

  size_t size() const { return size_; }

  void Swap(VertexRange& rhs) {
    std::swap(vl_, rhs.vl_);
    std::swap(v2idx_, rhs.v2idx_);
    std::swap(size_, rhs.size_);
  }

  const size_t begin_value() const { return 0; }

  const size_t end_value() const { return size_; }

  const size_t GetVertexLoc(const Vertex& v) const {
    return v2idx_->at(v.grin_v);
  }

 private:
  GRIN_GRAPH g_;
  GRIN_VERTEX_LIST vl_;
  size_t size_;
  const std::unordered_map<GRIN_VERTEX, size_t>* v2idx_;
};
#endif

template <typename T>
class VertexArray : public grape::Array<T, grape::Allocator<T>> {
  using Base = grape::Array<T, grape::Allocator<T>>;

 public:
  VertexArray() noexcept : Base(), fake_start_(NULL) {}
  explicit VertexArray(const VertexRange& range)
      : Base(range.size()), range_(range) {
    fake_start_ = Base::data() - range_.begin_value();
  }
  VertexArray(const VertexRange& range, const T& value)
      : Base(range.size(), value), range_(range) {
    fake_start_ = Base::data() - range_.begin_value();
  }

  ~VertexArray() = default;

  void Init(const VertexRange& range) {
    Base::clear();
    Base::resize(range.size());
    range_ = range;
    fake_start_ = Base::data() - range_.begin_value();
  }

  void Init(const VertexRange& range, const T& value) {
    Base::clear();
    Base::resize(range.size(), value);
    range_ = range;
    fake_start_ = Base::data() - range_.begin_value();
  }

  void SetValue(VertexRange& range, const T& value) {
    std::fill_n(&Base::data()[range.begin_value() - range_.begin_value()],
                range.size(), value);
  }
  void SetValue(const Vertex& loc, const T& value) {
    fake_start_[range_.GetVertexLoc(loc)] = value;
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](const Vertex& loc) {
    return fake_start_[range_.GetVertexLoc(loc)];
  }
  inline const T& operator[](const Vertex& loc) const {
    return fake_start_[range_.GetVertexLoc(loc)];
  }

  void Swap(VertexArray& rhs) {
    Base::swap((Base&) rhs);
    range_.Swap(rhs.range_);
    std::swap(fake_start_, rhs.fake_start_);
  }

  void Clear() {
    Base::clear();
    fake_start_ = NULL;
  }

  const VertexRange& GetVertexRange() const { return range_; }

 private:
  VertexRange range_;
  T* fake_start_;
};

/*
class DenseVertexSet {
  using range_t = VertexRange;
  using vertex_t = Vertex;
 public:
  DenseVertexSet() = default;

  explicit DenseVertexSet(const range_t& range)
      : bs_(range.size()), range_(range) {}

  ~DenseVertexSet() = default;

  void Init(const range_t& range, ThreadPool& thread_pool) {
    bs_.init(range.size());
    bs_.parallel_clear(thread_pool);
  }

  void Init(const range_t& range) {
    bs_.init(range.size());
    bs_.clear();
  }

  void Insert(const vertex_t& u) { bs_.set_bit(range_.GetVertexLoc(u)); }

  bool InsertWithRet(const vertex_t& u) {
    return bs_.set_bit_with_ret(range_.GetVertexLoc(u));
  }

  void Erase(const vertex_t& u) { bs_.reset_bit(range_.GetVertexLoc(u)); }

  bool EraseWithRet(const vertex_t& u) {
    return bs_.reset_bit_with_ret(range_.GetVertexLoc(u));
  }

  bool Exist(const vertex_t& u) const { return bs_.get_bit(range_.GetVertexLoc(u)); }

  const range_t& Range() const { return range_; }

  size_t Count() const { return bs_.count(); }

  size_t ParallelCount(ThreadPool& thread_pool) const {
    return bs_.parallel_count(thread_pool);
  }

  void Clear() { bs_.clear(); }

  void ParallelClear(ThreadPool& thread_pool) {
    bs_.parallel_clear(thread_pool);
  }

  void Swap(DenseVertexSet& rhs) {
    bs_.swap(rhs.bs_);
    range_.Swap(rhs.range_);
  }

  bool PartialEmpty(size_t beg, size_t end) const {
    return bs_.partial_empty(beg - range_.begin_value(), end - range_.begin_value());
  }

  grape::Bitset& GetBitset() { return bs_; }

  const grape::Bitset& GetBitset() const { return bs_; }

  bool Empty() const { return bs_.empty(); }

 private:
  grape::Bitset bs_;
  VertexRange range_;
};
*/

#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
template <typename T>
struct Nbr {
 public:
  Nbr() : g_{GRIN_NULL_GRAPH}, al_(GRIN_NULL_LIST), cur_(0) {}
  Nbr(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t cur) : g_(g), al_(al), cur_(cur) {}
  Nbr(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, size_t cur, const char* default_prop_name)
    : g_{g}, al_(al), cur_(cur), default_prop_name_(default_prop_name) {}
  Nbr(const Nbr& rhs) : g_(rhs.g_), al_(rhs.al_), cur_(rhs.cur_), default_prop_name_(rhs.default_prop_name_)  {}

  Nbr& operator=(const Nbr& rhs) {
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    default_prop_name_ = rhs.default_prop_name_;
    return *this;
  }

  Vertex neighbor() const {
    return Vertex(g_, grin_get_neighbor_from_adjacent_list(g_, al_, cur_));
  }

  Vertex get_neighbor() const {
    return Vertex(g_, grin_get_neighbor_from_adjacent_list(g_, al_, cur_));
  }

  // TODO: add a wrapper like vertex to care the destroy of edge
  GRIN_EDGE get_edge() const {
    return grin_get_edge_from_adjacent_list(g_, al_, cur_);
  }

  T get_data() const {
    auto _e = grin_get_edge_from_adjacent_list(g_, al_, cur_);
    auto type = grin_get_edge_type(g_, _e);
    auto prop = grin_get_edge_property_by_name(g_, type, default_prop_name_);
    auto _value = grin_get_edge_property_value_of_double(g_, _e, prop);
    grin_destroy_edge_property(g_, prop);
    grin_destroy_edge_type(g_, type);
    grin_destroy_edge(g_, _e);
    return _value;
  }

  inline Nbr& operator++() {
    cur_++;
    return *this;
  }

  inline Nbr operator++(int) {
    cur_++;
    return *this;
  }

  inline Nbr& operator--() {
    cur_--;
    return *this;
  }

  inline Nbr operator--(int) {
    cur_--;
    return *this;
  }

  inline bool operator==(const Nbr& rhs) const {
    return al_ == rhs.al_ && cur_ == rhs.cur_;
  }
  inline bool operator!=(const Nbr& rhs) const {
    return al_ != rhs.al_ || cur_ != rhs.cur_;
  }

  inline bool operator<(const Nbr& rhs) const {
    return al_ == rhs.al_ && cur_ < rhs.cur_;
  }

  inline const Nbr& operator*() const { return *this; }
  inline const Nbr* operator->() const { return this; }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST al_;
  size_t cur_;
  // const std::vector<GRIN_EDGE_PROPERTY_TABLE>* epts_;
  const char* default_prop_name_;
};
#else
template <typename T>
struct Nbr {
 public:
  Nbr() : g_(GRIN_NULL_GRAPH), al_(GRIN_NULL_LIST), cur_(GRIN_NULL_LIST_ITERATOR) {}
  Nbr(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, GRIN_ADJACENT_LIST_ITERATOR cur, const char* default_prop_name)
    : g_{g}, al_(al), cur_(cur), default_prop_name_(default_prop_name) {}
  Nbr(const Nbr& rhs) = delete;
  Nbr(Nbr&& rhs) {  // move constructor
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    default_prop_name_ = rhs.default_prop_name_;
    rhs.cur_ = GRIN_NULL_LIST_ITERATOR;
  }
  // Nbr(Nbr& rhs) : g_{rhs.g_}, al_(rhs.al_), cur_(rhs.cur_), default_prop_name_(rhs.default_prop_name_)  {}
  ~Nbr() {
    grin_destroy_adjacent_list_iter(g_, cur_);
  }

  inline const Nbr& operator*() const { return *this; }

  inline const Nbr& operator->() const { return this; }

  Nbr& operator=(const Nbr& rhs) = delete;

  inline Nbr& operator=(Nbr&& rhs) {
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    default_prop_name_ = rhs.default_prop_name_;
    rhs.cur_ = nullptr;
    return *this;
  }

  Vertex neighbor() {
    return Vertex(g_, grin_get_neighbor_from_adjacent_list_iter(g_, cur_));
  }

  Vertex get_neighbor() {
    return Vertex(g_, grin_get_neighbor_from_adjacent_list_iter(g_, cur_));
  }

  GRIN_EDGE get_edge() {
    return grin_get_edge_from_adjacent_list_iter(g_, cur_);
  }

  T get_data() const {
    auto _e = grin_get_edge_from_adjacent_list_iter(g_, cur_);
    auto type = grin_get_edge_type(g_, _e);
    auto prop = grin_get_edge_property_by_name(g_, type, default_prop_name_);
    return grin_get_edge_property_value_of_double(g_, _e, prop);
  }

  inline Nbr& operator++() {
    grin_get_next_adjacent_list_iter(g_, cur_);
    return *this;
  }

  inline bool is_end() const {
    return grin_is_adjacent_list_end(g_, cur_);
  }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST al_;
  GRIN_ADJACENT_LIST_ITERATOR cur_;
  const char* default_prop_name_;
};
#endif


#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
template <typename T>
class AdjList {
  using nbr_t = Nbr<T>;

 public:
  AdjList(): g_(GRIN_NULL_GRAPH), adj_list_(GRIN_NULL_LIST), begin_(0), end_(0) {}
  AdjList(GRIN_GRAPH g, GRIN_ADJACENT_LIST adj_list,
          const char* prop_name, size_t begin, size_t end)
    : g_{g}, adj_list_(adj_list), prop_name_(prop_name), begin_(begin), end_(end) {}
  AdjList(const AdjList&) = delete;  // disable copy constructor
  void operator=(const AdjList&) = delete;  // disable copy assignment
  AdjList(AdjList&& rhs) = delete;  // disable move constructor

  ~AdjList() {
    // grin_destroy_adjacent_list(g_, adj_list_);
  }

  inline nbr_t begin() const {
    return nbr_t(g_, adj_list_, begin_, prop_name_);
  }

  inline nbr_t end() const {
    return nbr_t(g_, adj_list_, end_, prop_name_);
  }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return begin_ < end_; }

  size_t size() const { return end_ - begin_; }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST adj_list_;
  const char* prop_name_;
  size_t begin_;
  size_t end_;
};
#else
template <typename T>
class AdjList {
  using nbr_t = Nbr<T>;

 public:
  AdjList(): g_(GRIN_NULL_GRAPH), adj_list_(GRIN_NULL_LIST) {}
  AdjList(GRIN_GRAPH g, GRIN_ADJACENT_LIST adj_list, const char* prop_name)
    : g_{g}, adj_list_(adj_list), prop_name_(prop_name) {}
  AdjList(const AdjList&) = delete;  // disable copy constructor
  void operator=(const AdjList&) = delete;  // disable copy assignment
  AdjList(AdjList&& rhs) = delete;  // disable move constructor

  ~AdjList() {
    // grin_destroy_adjacent_list(g_, adj_list_);
  }

  inline nbr_t begin() const {
    auto iter = grin_get_adjacent_list_begin(g_, adj_list_);
    return nbr_t(g_, adj_list_, iter, prop_name_);
  }

  inline bool IsEnd(const nbr_t& iter) {
    return iter.is_end();
  }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST adj_list_;
  const char* prop_name_;
};
#endif
}  // namespace arrow_flattened_fragment_grin_impl

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
  using ref_t = int64_t;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = arrow_flattened_fragment_grin_impl::Vertex;
  using fid_t = GRIN_PARTITION_ID;
  using vertex_range_t = arrow_flattened_fragment_grin_impl::VertexRange;
  using inner_vertices_t = vertex_range_t;
  using outer_vertices_t = vertex_range_t;
  using vertices_t = vertex_range_t;

  using adj_list_t =
      arrow_flattened_fragment_grin_impl::AdjList<EDATA_T>;

  template <typename DATA_T>
  using vertex_array_t = arrow_flattened_fragment_grin_impl::VertexArray<DATA_T>;

  template <typename DATA_T>
  using inner_vertex_array_t = arrow_flattened_fragment_grin_impl::VertexArray<DATA_T>;

  template <typename DATA_T>
  using outer_vertex_array_t = arrow_flattened_fragment_grin_impl::VertexArray<DATA_T>;

  using v2index_t = std::unordered_map<GRIN_VERTEX, size_t>;

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
    tvl_ = grin_get_vertex_list_by_type(g_, vt);
    ivl_ = grin_get_vertex_list_by_type_select_master(g_, vt);
    ovl_ = grin_get_vertex_list_by_type_select_mirror(g_, vt);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    tvnum_ = grin_get_vertex_list_size(g_, tvl_);
    ivnum_ = grin_get_vertex_list_size(g_, ivl_);
    ovnum_ = grin_get_vertex_list_size(g_, ovl_);

    auto et = grin_get_edge_type_from_list(g_, etl_, 0);
    auto inner_verices = this->InnerVertices();
    for (const auto& v : inner_verices) {
      v2iadj_[v.grin_v] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v.grin_v, et);
      v2oadj_[v.grin_v] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v.grin_v, et);
    }
#elif defined(GRIN_ENABLE_VERTEX_LIST_ITERATOR)
    auto iter = grin_get_vertex_list_begin(g_, tvl_);
    tv2i_ = std::make_shared<v2index_t>();
    size_t index = 0;
    while (!grin_is_vertex_list_end(g_, iter)) {
      auto v = grin_get_vertex_from_iter(g_, iter);
      (*tv2i_)[v] = index;
      ++index;
      grin_destroy_vertex(g_, v);
      grin_get_next_vertex_list_iter(g_, iter);
    }
    grin_destroy_vertex_list_iter(g_, iter);
    index = 0;
    auto iv_iter = grin_get_vertex_list_begin(g_, ivl_);
    iv2i_ = std::make_shared<v2index_t>();
    auto et = grin_get_edge_type_from_list(g_, etl_, 0);
    while (!grin_is_vertex_list_end(g_, iv_iter)) {
        auto v = grin_get_vertex_from_iter(g_, iv_iter);
        (*iv2i_)[v] = index;
        ++index;
        ++ivnum_;
        v2iadj_[v] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::IN, v, et);
        v2oadj_[v] = grin_get_adjacent_list_by_edge_type(g_, GRIN_DIRECTION::OUT, v, et);
        grin_destroy_vertex(g_, v);
        grin_get_next_vertex_list_iter(g_, iv_iter);
    }
    grin_destroy_vertex_list_iter(g_, iv_iter);
    auto ov_iter = grin_get_vertex_list_begin(g_, ovl_);
    ov2i_ = std::make_shared<v2index_t>();
    index = 0;
    while (!grin_is_vertex_list_end(g_, ov_iter)) {
      auto v = grin_get_vertex_from_iter(g_, ov_iter);
      (*ov2i_)[v] = index;
      ++index;
      ++ovnum_;
      grin_destroy_vertex(g_, v);
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
#else  // simple graph storage
  explicit GRINFlattenedFragment(GRIN_PARTITIONED_GRAPH partitioned_graph, GRIN_PARTITION partition)
    : pg_(partitioned_graph), partition_(partition) {
    g_ = grin_get_local_graph_by_partition(pg_, partition_);
    tvnum_ = ivnum_ = ovnum_ = 0;
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    tvnum_ = grin_get_vertex_list_size(g_, tvl_);
    ivnum_ = grin_get_vertex_list_size(g_, ivl_);
    ovnum_ = grin_get_vertex_list_size(g_, vl2);
#elif defined(GRIN_ENABLE_VERTEX_LIST_ITERATOR)
    auto iter = grin_get_vertex_list_begin(g_, tvl_);
    while (!grin_is_vertex_list_end(g_, iter)) {
      ++tvnum_;
      grin_get_next_vertex_list_iter(g_, iter);
    }
    auto iv_iter = grin_get_vertex_list_begin(g_, ivl_);
    while (!grin_is_vertex_list_end(g_, iv_iter)) {
      ++ivnum_;
      grin_get_next_vertex_list_iter(g_, iv_iter);
    }
    auto ov_iter = grin_get_vertex_list_begin(g_, ovl_);
    while (!grin_is_vertex_list_end(g_, ov_iter)) {
      ++ovnum_;
      grin_get_next_vertex_list_iter(g_, ov_iter);
    }
    grin_destroy_vertex_list_iter(g_, iter);
    grin_destroy_vertex_list_iter(g_, iv_iter);
    grin_destroy_vertex_list_iter(g_, ov_iter);
#endif
  }
#endif

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
    return vertex_range_t(g_, tvl_, tv2i_.get());
#endif
  }

  inline vertex_range_t InnerVertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, ivl_, 0, ivnum_);
#else
    return vertex_range_t(g_, ivl_, iv2i_.get());
#endif
  }

  inline vertex_range_t OuterVertices() const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return vertex_range_t(g_, ovl_, 0, ovnum_);
#else
    return vertex_range_t(g_, ovl_, ov2i_.get());
#endif
  }

#ifdef GRIN_ENABLE_VERTEX_ORIGINAL_ID_OF_INT64
  bool GetVertex(oid_t& oid, vertex_t& v) const {
    auto grin_v = grin_get_vertex_by_original_id_of_int64(g_, oid);
    if (grin_v == GRIN_NULL_VERTEX) {
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  bool GetInnerVertex(oid_t& oid, vertex_t& v) const {
    auto grin_v = grin_get_vertex_by_original_id_of_int64(g_, oid);
    if (grin_v == GRIN_NULL_VERTEX || grin_is_mirror_vertex(g_, grin_v)) {
      grin_destroy_vertex(g_, grin_v);
      return false;
    }
    v.Refresh(g_, grin_v);
    return true;
  }

  inline bool GetId(const vertex_t& v, oid_t& oid) const {
    // if (GRIN_DATATYPE_ENUM<oid_t>::value != grin_get_vertex_original_id_datatype(g_)) return false;
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
    oid = std::move(grin_get_vertex_original_id_of_int64(g_, v.grin_v));
    return true;
  }

  inline oid_t GetId(const vertex_t& v) const {
    return grin_get_vertex_original_id_of_int64(g_, v.grin_v);
  }

/*
  inline oid_t Gid2Oid(const ref_t& gid) const {
    return fragment_->Gid2Oid(gid);
  }

  inline bool Oid2Gid(const oid_t& oid, ref_t& gid) const {
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
    auto vref = grin_get_vertex_ref_by_vertex(g_, u.grin_v);
    auto partition = grin_get_master_partition_from_vertex_ref(g_, vref);
    auto fid = grin_get_partition_id(pg_, partition);
    grin_destroy_vertex_ref(g_, vref);
    grin_destroy_partition(g_, partition);
    return fid;
  }

  inline bool Gid2Vertex(const ref_t& ref, vertex_t& v) const {
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

  inline ref_t GetInnerVertexGid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline ref_t GetOuterVertexGid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  inline const ref_t Vertex2Gid(const vertex_t& v) const {
    auto ref = grin_get_vertex_ref_by_vertex(g_, v.grin_v);
    return grin_serialize_vertex_ref_as_int64(g_, ref);
  }

  bool GetData(const vertex_t& v, vdata_t& value) const {
    if (v.grin_v == GRIN_NULL_VERTEX) return false;
#ifdef GRIN_WITH_VERTEX_DATA
    // if (GRIN_DATATYPE_ENUM<vdata_t>::value != grin_get_vertex_data_type(g_, v)) return false;
    value = grin_get_vertex_property_value_of_int64(g_, v.grin_v);
    return true;
#else
    auto vtype = grin_get_vertex_type(g_, v.grin_v);
    auto v_prop = grin_get_vertex_property_by_name(g_, vtype, v_prop_.data());
    bool succ = false;
    if (v_prop != GRIN_NULL_VERTEX_PROPERTY) {
      // if (GRIN_DATATYPE_ENUM<vdata_t>::value != grin_get_vertex_property_datatype(g_, v_prop)) return false;
      value = grin_get_vertex_property_value_of_int64(g_, v.grin_v, v_prop);
      succ = true;
    }
    grin_destroy_vertex_type(g_, vtype);
    grin_destroy_vertex_property(g_, v_prop);
    return succ;
#endif
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
    auto al = v2oadj_.at(v.grin_v);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto sz = grin_get_adjacent_list_size(g_, al);
    return adj_list_t(g_, al, e_prop_.c_str(), 0, sz);
#else
    return adj_list_t(g_, al, e_prop_.c_str());
#endif
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v) const {
    auto al = v2iadj_.at(v.grin_v);

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
    auto al = v2oadj_.at(v.grin_v);
    // auto al = grin_get_adjacent_list(g_, GRIN_DIRECTION::OUT, v.grin_v);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    return static_cast<int>(grin_get_adjacent_list_size(g_, al));
    // grin_destroy_adjacent_list(g_, al);
#else
    int degree = 0;
    auto iter = grin_get_adjacent_list_begin(g_, al);
    while (!grin_is_adjacent_list_end(g_, iter)) {
      ++degree;
      grin_get_next_adjacent_list_iter(g_, iter);
    }
    grin_destroy_adjacent_list_iter(g_, iter);
    // grin_destroy_adjacent_list(g_, al);
    return degree;
#endif
  }

  inline int GetLocalInDegree(const vertex_t& v) const {
    auto al = v2iadj_.at(v.grin_v);
    // auto al = grin_get_adjacent_list(g_, GRIN_DIRECTION::IN, v.grin_v);

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
    // grin_destroy_adjacent_list(g_, al);
    return degree;
#endif
  }

  inline grape::DestList IEDests(const vertex_t& v) const {
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    auto pos = grin_get_position_of_vertex_from_sorted_list(g_, ivl_, v.grin_v);
    return grape::DestList(idoffset_[pos],
                           idoffset_[pos + 1]);
#else
    auto oid = GetId(v);
    auto pos = iv2i_->at(oid);
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
    auto oid = GetId(v);
    auto pos = iv2i_->at(oid);
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
    auto oid = GetId(v);
    auto pos = iv2i_->at(oid);
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
        if (in_edge) {
          auto al = v2iadj_.at(v);
          auto sz = grin_get_adjacent_list_size(g_, al);
          for (size_t j = 0; j < sz; ++j) {
            auto neighbor = grin_get_neighbor_from_adjacent_list(g_, al, j);
            auto vref = grin_get_vertex_ref_by_vertex(g_, neighbor);
            auto p = grin_get_master_partition_from_vertex_ref(g_, vref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
          }
        }
        if (out_edge) {
          auto al = v2oadj_.at(v);
          auto sz = grin_get_adjacent_list_size(g_, al);
          for (size_t j = 0; j < sz; ++j) {
            auto neighbor = grin_get_neighbor_from_adjacent_list(g_, al, j);
            auto vref = grin_get_vertex_ref_by_vertex(g_, neighbor);
            auto p = grin_get_master_partition_from_vertex_ref(g_, vref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
          }
        }
        id_num[i] = dstset.size();
        for (auto fid : dstset) {
          fid_list.push_back(fid);
        }
        ++v;
      }

      fid_list.shrink_to_fit();
      fid_list_offset[0] = fid_list.data();
      for (size_t i = 0; i < ivnum_; ++i) {
        fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
      }
#else  // GRIN_ENABLE_VERTEX_LIST_ARRAY
      auto inner_vertices = InnerVertices();

      if (!fid_list_offset.empty()) {
        return;
      }
      fid_list_offset.resize(ivnum_ + 1, NULL);
      std::vector<int> id_num(ivnum_, 0);
      std::set<fid_t> dstset;
      auto iter = inner_vertices.begin();
      size_t index = 0;
      while (!iter.is_end()) {
        dstset.clear();
        auto v = *iter;
        if (in_edge) {
          auto es = GetIncomingAdjList(v);
          auto e_iter = es.begin();
          while (!e_iter.is_end()) {
            auto neighbor = e_iter.neighbor();
            auto vref = grin_get_vertex_ref_by_vertex(g_, neighbor.grin_v);
            auto p = grin_get_master_partition_from_vertex_ref(g_, vref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
            ++e_iter;
          }
        }
        if (out_edge) {
          auto es = GetOutgoingAdjList(v);
          auto e_iter = es.begin();
          while (!e_iter.is_end()) {
            auto neighbor = e_iter.neighbor();
            auto vref = grin_get_vertex_ref_by_vertex(g_, neighbor.grin_v);
            auto p = grin_get_master_partition_from_vertex_ref(g_, vref);

            if (!grin_equal_partition(g_, p, partition_)) {
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
              dstset.insert(grin_get_partition_id(g_, p));
#else
              // todo
#endif
            }
            ++e_iter;
          }
        }
        id_num[index] = dstset.size();
        for (auto fid : dstset) {
          fid_list.push_back(fid);
        }
        ++index;
        ++iter;
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
  fid_t fid_;
  fid_t fnum_;

  std::vector<fid_t> idst_, odst_, iodst_;
  std::vector<fid_t*> idoffset_, odoffset_,
      iodoffset_;
  GRIN_EDGE_TYPE_LIST etl_;
  std::unordered_map<GRIN_VERTEX, GRIN_ADJACENT_LIST> v2iadj_;
  std::unordered_map<GRIN_VERTEX, GRIN_ADJACENT_LIST> v2oadj_;

  GRIN_VERTEX_LIST ivl_, ovl_, tvl_;
  GRIN_VERTEX_TYPE_LIST vtl_;
  std::shared_ptr<v2index_t> iv2i_;
  std::shared_ptr<v2index_t> ov2i_;
  std::shared_ptr<v2index_t> tv2i_;

  // std::shared_ptr<std::vector<GRIN_EDGE_PROPERTY_TABLE>> epts_;

  size_t ivnum_;
  size_t ovnum_;
  size_t tvnum_;
};

}  // namespace gs
#endif  // ANALYTICAL_ENGINE_CORE_FRAGMENT_ARROW_FLATTENED_FRAGMENT_H_
