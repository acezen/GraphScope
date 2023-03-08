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

#ifndef ANALYTICAL_ENGINE_APPS_FLASH_CONTEXT_H_
#define ANALYTICAL_ENGINE_APPS_FLASH_CONTEXT_H_

namespace gs {

#include "grape/grape.h"

template <typename FRAG_T, typename VALUE_T, typename RESULT_T>
class FlashVertexDataContext : public grape::VertexDataContext<FRAG_T, RESULT_T> {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using result_t = RESULT_T;

 public:
  explicit FlashVertexDataContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, RESULT_T>(fragment, true),
        result(this->data()) {}

  void SetResult(FlashWare<FRAG_T, VALUE_T>* fw) {
    // TODO: parallelize this
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();
    for (auto v : inner_vertices) {
      this->result[v] = fw->Get(fw->Lid2Key(v.GetValue()))->res;
    }
  }

  typename FRAG_T::template vertex_array_t<result_t>& result;
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_FLASH_CONTEXT_H_
