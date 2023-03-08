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

#ifndef ANALYTICAL_ENGINE_APPS_FLASH_RANKING_PAGERANK_H_
#define ANALYTICAL_ENGINE_APPS_FLASH_RANKING_PAGERANK_H_

#include "grape/grape.h"

#include "apps/flash/api.h"
#include "apps/flash/flash_app_base.h"
#include "apps/flash/flash_context.h"
#include "apps/flash/value_type.h"
#include "apps/flash/flash_worker.h"

namespace gs {

template <typename FRAG_T>
class PRFlash : public FlashAppBase<FRAG_T, PR_TYPE> {
 public:
  INSTALL_FLASH_WORKER(PRFlash<FRAG_T>, PR_TYPE, FRAG_T)
  using context_t = FlashVertexDataContext<FRAG_T, PR_TYPE, double>;

  bool sync_all_ = false;

  double* Res(value_t* v) { return &(v->res); };

  void Run(const fragment_t& graph, const  int max_iters, double damping = 0.85) {
    Print("Run PageRank with Flash, max_iters = %d, damping = %f\n", max_iters, damping);
    int n_vertex = graph.GetTotalVerticesNum();
    Print("Total vertices: %d\n", n_vertex);

    DefineMapV(init_v) {
      v.res = 1.0 / n_vertex;
      v.next = 0;
      v.deg = OutDeg(id);
    };
    VertexMap(All, CTrueV, init_v);
    // for (auto &i:All.s) LOG(INFO) << "Init: " << i <<' '<< GetV(i)->res;
    Print("Init complete\n");

    DefineMapE(update) {
      d.next += damping * s.res / s.deg;
    };
    DefineMapV(local) {
      v.res = v.next + (1 - damping) / n_vertex +
              (v.deg == 0 ? damping * v.res : 0);
      v.next = 0;
    };

    for (int i = 0; i < max_iters; i++) {
      Print("Round %d\n", i);
      EdgeMapDense(All, ED, CTrueE, update, CTrueV, false);
      VertexMap(All, CTrueV, local);
    }
  }
};

}  // namespace gs

#endif  // ANALYTICAL_ENGINE_APPS_FLASH_RANKING_PAGERANK_H_
