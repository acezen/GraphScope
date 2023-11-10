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

#ifndef ANALYTICAL_ENGINE_CORE_GRIN_GRIN_UTIL_H_
#define ANALYTICAL_ENGINE_CORE_GRIN_GRIN_UTIL_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vineyard/common/util/static_if.h"
#include "vineyard/graph/grin/predefine.h"
// #include "interfaces/grin/predefine.h"

#include "grin/topology/structure.h"
#include "grin/topology/vertexlist.h"
#include "grin/topology/adjacentlist.h"

#include "grin/partition/partition.h"
#include "grin/partition/topology.h"
#include "grin/partition/reference.h"

#include "grin/property/type.h"
#include "grin/property/property.h"
#include "grin/property/propertylist.h"
#include "grin/property/topology.h"
#include "grin/property/value.h"

#include "grin/index/order.h"
#include "grin/index/external_id.h"
#include "grin/index/internal_id.h"

namespace gs {

namespace grin_util {

// A wrapper for GRIN_VERTEX
struct Vertex {
  Vertex(uint64_t a) noexcept : g_(GRIN_NULL_GRAPH), grin_v(GRIN_NULL_VERTEX) {}  // compatible with grape
  Vertex() noexcept : g_(GRIN_NULL_GRAPH), grin_v(GRIN_NULL_VERTEX) {}
  explicit Vertex(GRIN_GRAPH g, GRIN_VERTEX v) noexcept : g_(g), grin_v(v) {}
  inline Vertex& operator=(Vertex&& rhs) noexcept {
    g_ = rhs.g_;
    grin_v = rhs.grin_v;
    rhs.grin_v = GRIN_NULL_VERTEX;
    return *this;
  }
  Vertex(const Vertex& rhs) : g_(rhs.g_), grin_v(rhs.grin_v) {
    /*
    grin_v = rhs.grin_v;
    auto v_ref = grin_get_vertex_ref_by_vertex(g_, rhs.grin_v);
    grin_v = grin_get_vertex_from_vertex_ref(g_, v_ref);
    grin_destroy_vertex_ref(g_, v_ref);
    */
  }
  Vertex(Vertex&& rhs) noexcept : g_(rhs.g_), grin_v(rhs.grin_v) {
    rhs.grin_v = GRIN_NULL_VERTEX;
  }
  inline Vertex& operator=(const Vertex& rhs) {
    g_ = rhs.g_;
    grin_v = rhs.grin_v;
    /*
    grin_v = GRIN_NULL_VERTEX;
    auto v_ref = grin_get_vertex_ref_by_vertex(g_, rhs.grin_v);
    grin_v = grin_get_vertex_from_vertex_ref(g_, v_ref);
    grin_destroy_vertex_ref(g_, v_ref);
    */
    return *this;
  }
  ~Vertex() {
    grin_destroy_vertex(g_, grin_v);
  }
  inline bool operator<(const Vertex& rhs) const {
    return grin_v < rhs.grin_v;
  }

  inline void Refresh(GRIN_GRAPH g, GRIN_VERTEX v) {
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
  VertexRange() noexcept : g_(GRIN_NULL_GRAPH), vl_(GRIN_NULL_VERTEX_LIST), vt_(GRIN_NULL_VERTEX_TYPE), begin_(0), end_(0) {}
  VertexRange(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, GRIN_VERTEX_TYPE vt, const size_t begin, const size_t end)
      noexcept : g_(g), vl_(vl), vt_(vt), begin_(begin), end_(end) {}
  VertexRange(const VertexRange& r) noexcept : g_(r.g_), vl_(r.vl_), vt_(r.vt_), begin_(r.begin_), end_(r.end_) {}

  ~VertexRange() = default;

  inline VertexRange& operator=(const VertexRange& r) noexcept {
    g_ = r.g_;
    vl_ = r.vl_;
    vt_ = r.vt_;
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
    iterator() noexcept : g_(GRIN_NULL_GRAPH), vl_(GRIN_NULL_VERTEX_LIST), cur_(0) {}
    explicit iterator(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t idx) noexcept : g_(g), vl_(vl), cur_(idx) {}
    iterator(const iterator& rhs) = default;
    inline iterator& operator=(const iterator& rhs) = default;
    ~iterator() = default;
    reference_type operator*() noexcept {
      return Vertex(g_, grin_get_vertex_from_list(g_, vl_, cur_));
    }

    inline iterator& operator++() noexcept {
      ++cur_;
      return *this;
    }

    inline iterator operator++(int) noexcept {
      return iterator(g_, vl_, cur_ + 1);
    }

    inline iterator& operator--() noexcept {
      --cur_;
      return *this;
    }

    inline iterator operator--(int) noexcept {
      return iterator(g_, vl_, cur_--);
    }

    inline iterator operator+(size_t offset) const noexcept {
      return iterator(g_, vl_, cur_ + offset);
    }

    inline bool operator==(const iterator& rhs) const noexcept {
      return cur_ == rhs.cur_;
    }

    inline bool operator!=(const iterator& rhs) const noexcept {
      return cur_ != rhs.cur_;
    }

    inline bool operator<(const iterator& rhs) const noexcept {
      return vl_ == rhs.vl_ && cur_ < rhs.cur_;
    }
  };

  inline iterator begin() const { return iterator(g_, vl_, begin_); }

  inline iterator end() const { return iterator(g_, vl_, end_); }

  inline size_t size() const { return end_ - begin_; }

  inline void Swap(VertexRange& rhs) {
    std::swap(vl_, rhs.vl_);
    std::swap(vt_, rhs.vt_);
    std::swap(begin_, rhs.begin_);
    std::swap(end_, rhs.end_);
  }

  inline void SetRange(const size_t begin, const size_t end) {
    begin_ = begin;
    end_ = end;
  }

  inline size_t begin_value() const { return begin_; }

  inline size_t end_value() const { return end_; }

  inline int64_t GetVertexLoc(const Vertex& v) const {
    return grin_get_vertex_internal_id_by_type(g_, vt_, v.grin_v);
  }
 public:
  GRIN_GRAPH g_;
  GRIN_VERTEX_LIST vl_;
  GRIN_VERTEX_TYPE vt_;
  size_t begin_;
  size_t end_;
};
#else
class VertexRange { // TODO(wanglei): add vtype as member
 public:
  VertexRange() noexcept : g_(GRIN_NULL_GRAPH), vl_(GRIN_NULL_VERTEX_LIST), size_(0) {}
  VertexRange(GRIN_GRAPH g, GRIN_VERTEX_LIST vl, size_t size)
      : g_(g), vl_(vl), size_(size) {}

  VertexRange(const VertexRange& r) : g_(r.g_), vl_(r.vl_), size_(r.size_) {}

  ~VertexRange() = default;

  inline VertexRange& operator=(const VertexRange& r) {
    g_ = r.g_;
    vl_ = r.vl_;
    size_ = r.size_;
    return *this;
  }

  class iterator {
    using reference_type = Vertex;

   private:
    GRIN_GRAPH g_;
    GRIN_VERTEX_LIST_ITERATOR cur_;

   public:
    iterator() noexcept : g_(GRIN_NULL_GRAPH), cur_(GRIN_NULL_VERTEX_LIST_ITERATOR) {}
    explicit iterator(GRIN_GRAPH g, GRIN_VERTEX_LIST_ITERATOR iter) noexcept : g_(g), cur_(iter) {}
    iterator(const iterator& rhs) = delete;
    iterator& operator=(const iterator& rhs) = delete;
    iterator& operator=(iterator&& rhs) {
      g_ = rhs.g_;
      cur_ = rhs.cur_;
      rhs.cur_ = GRIN_NULL_VERTEX_LIST_ITERATOR;
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

  size_t size() const { 
    return size_; 
  }

  void Swap(VertexRange& rhs) {
    std::swap(vl_, rhs.vl_);
    std::swap(size_, rhs.size_);
  }

  const size_t begin_value() const { return 0; }

  const size_t end_value() const { return size_; }

  int64_t GetVertexLoc(const Vertex& v) const {
     auto vt = grin_get_vertex_type(g_, v.grin_v);
    return grin_get_vertex_internal_id_by_type(g_, vt, v.grin_v);
  }

 private:
  GRIN_GRAPH g_;
  GRIN_VERTEX_LIST vl_;
  size_t size_;
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
    // auto internal_id = range_.GetVertexLoc(loc);
    fake_start_[loc.grin_v] = value;
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](const Vertex& loc) {
    auto internal_id = range_.GetVertexLoc(loc);
    return fake_start_[loc.grin_v];
  }
  inline const T& operator[](const Vertex& loc) const {
    // auto internal_id = range_.GetVertexLoc(loc);
    return fake_start_[loc.grin_v];
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
  Nbr() : g_{GRIN_NULL_GRAPH}, al_(NULL), cur_(0) {}
  Nbr(GRIN_GRAPH g, const GRIN_ADJACENT_LIST* al, size_t cur, GRIN_EDGE_PROPERTY prop)
    : g_{g}, al_(al), cur_(cur), prop_(prop) {}
  Nbr(const Nbr& rhs) : g_(rhs.g_), al_(rhs.al_), cur_(rhs.cur_), prop_(rhs.prop_)  {}

  Nbr& operator=(const Nbr& rhs) {
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    prop_ = rhs.prop_;
    return *this;
  }

  Vertex neighbor() const {
    return Vertex(g_, grin_get_neighbor_from_adjacent_list(g_, *al_, cur_));
  }

  Vertex get_neighbor() const {
    return Vertex(g_, grin_get_neighbor_from_adjacent_list(g_, *al_, cur_));
  }

  // TODO: add a wrapper like vertex to care the destroy of edge
  GRIN_EDGE get_edge() const {
    return grin_get_edge_from_adjacent_list(g_, *al_, cur_);
  }

  T get_data() const {
    auto _e = grin_get_edge_from_adjacent_list(g_, *al_, cur_);
    auto type = grin_get_edge_type(g_, _e);
    T _value;
    vineyard::static_if<std::is_same<T, int64_t>{}>(
        [&](auto& _value) {
          _value = grin_get_edge_property_value_of_int64(g_, _e, prop_);
        })(_value);
    vineyard::static_if<std::is_same<T, double>{}>(
        [&](auto& _value) {
          _value = grin_get_edge_property_value_of_double(g_, _e, prop_);
        })(_value);
    grin_destroy_edge_property(g_, prop_);
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
    return cur_ == rhs.cur_;
  }
  inline bool operator!=(const Nbr& rhs) const {
    return cur_ != rhs.cur_;
  }

  inline bool operator<(const Nbr& rhs) const {
    return cur_ < rhs.cur_;
  }

  inline const Nbr& operator*() const { return *this; }
  inline const Nbr* operator->() const { return this; }

 private:
  GRIN_GRAPH g_;
  const GRIN_ADJACENT_LIST* al_;
  size_t cur_;
  GRIN_EDGE_PROPERTY prop_;
};
#else
template <typename T>
struct Nbr {
 public:
  Nbr() : g_(GRIN_NULL_GRAPH), al_(GRIN_NULL_ADJACENT_LIST), cur_(GRIN_NULL_ADJACENT_LIST_ITERATOR) {}
  Nbr(GRIN_GRAPH g, GRIN_ADJACENT_LIST al, GRIN_ADJACENT_LIST_ITERATOR cur, const char* default_prop_name)
    : g_{g}, al_(al), cur_(cur), default_prop_name_(default_prop_name) {}
  Nbr(const Nbr& rhs) = delete;
  Nbr(Nbr&& rhs) {  // move constructor
    g_ = rhs.g_;
    al_ = rhs.al_;
    cur_ = rhs.cur_;
    default_prop_name_ = rhs.default_prop_name_;
    rhs.cur_ = GRIN_NULL_ADJACENT_LIST_ITERATOR;
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
    /*
    auto _e = grin_get_edge_from_adjacent_list_iter(g_, cur_);
    auto type = grin_get_edge_type(g_, _e);
    auto prop = grin_get_edge_property_by_name(g_, type, default_prop_name_);
    return grin_get_edge_property_value_of_double(g_, _e, prop);
    */
    return 0;
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
  AdjList(): g_(GRIN_NULL_GRAPH), adj_list_(GRIN_NULL_ADJACENT_LIST), ep_(GRIN_NULL_EDGE_PROPERTY), begin_(0), end_(0) {}
  AdjList(GRIN_GRAPH g, GRIN_ADJACENT_LIST adj_list,
          GRIN_EDGE_PROPERTY property, size_t begin, size_t end)
    : g_{g}, adj_list_(adj_list), ep_(property), begin_(begin), end_(end) {}
  AdjList(const AdjList&) = delete;  // disable copy constructor
  void operator=(const AdjList&) = delete;  // disable copy assignment
  AdjList(AdjList&& rhs) = delete;  // disable move constructor

  ~AdjList() = default;

  inline nbr_t begin() const {
    return nbr_t(g_, &adj_list_, begin_, ep_);
  }

  inline nbr_t end() const {
    return nbr_t(g_, &adj_list_, end_, ep_);
  }

  inline size_t Size() const { return end_ - begin_; }

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return begin_ < end_; }

  size_t size() const { return end_ - begin_; }

 private:
  GRIN_GRAPH g_;
  GRIN_ADJACENT_LIST adj_list_;
  GRIN_EDGE_PROPERTY ep_;
  size_t begin_;
  size_t end_;
};
#else
template <typename T>
class AdjList {
  using nbr_t = Nbr<T>;

 public:
  AdjList(): g_(GRIN_NULL_GRAPH), adj_list_(GRIN_NULL_ADJACENT_LIST) {}
  AdjList(GRIN_GRAPH g, GRIN_ADJACENT_LIST adj_list, const char* prop_name)
    : g_{g}, adj_list_(adj_list), prop_name_(prop_name) {}
  AdjList(const AdjList&) = delete;  // disable copy constructor
  void operator=(const AdjList&) = delete;  // disable copy assignment
  AdjList(AdjList&& rhs) = delete;  // disable move constructor

  ~AdjList() {
    grin_destroy_adjacent_list(g_, adj_list_);
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
} // namespace grin_util
} // namespace gs
#endif // ANALYTICAL_ENGINE_CORE_GRIN_GRIN_UTIL_H_