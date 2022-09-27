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

#include <cstdio>
#include <fstream>
#include <string>

#include "glog/logging.h"


#include "grape/grape.h"
#include "grape/util.h"
#include "vineyard/client/client.h"
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/loader/arrow_fragment_builder.h"

#include "core/fragment/arrow_projected_fragment.h"

namespace bl = boost::leaf;

using FragmentType =
    gs::ArrowProjectedFragment<int64_t, uint64_t, int64_t, double>;
using GraphType =
      vineyard::ArrowFragment<vineyard::property_graph_types::OID_TYPE,
                              vineyard::property_graph_types::VID_TYPE>;
void RunPropertySSSP(std::shared_ptr<GraphType> fragment,
                     const grape::CommSpec& comm_spec,
                     const std::string& out_prefix) {
  using AppType = gs::PropertySSSP<GraphType>;
  auto app = std::make_shared<AppType>();
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();

  worker->Init(comm_spec, spec);
  MPI_Barrier(comm_spec.comm());
  worker->Query(4);

  auto ctx = worker->GetContext();

  worker->Finalize();
  std::shared_ptr<gs::LabeledVertexPropertyContextWrapper<GraphType>> t_ctx =
      std::dynamic_pointer_cast<
          gs::LabeledVertexPropertyContextWrapper<GraphType>>(ctx);
  MPI_Barrier(comm_spec.comm());
  {
    auto selector = gs::LabeledSelector::parse("r:label0.property0").value();
    auto arc = t_ctx->ToNdArray(comm_spec, selector, {"", ""}).value();
    if (comm_spec.fid() == 0) {
      gs::output_nd_array(std::move(*arc), out_prefix + "nd_array_result_0");
    } else {
      CHECK(arc->Empty());
    }
  }
}

void Run(vineyard::Client& client, const grape::CommSpec& comm_spec,
         vineyard::ObjectID id) {
  std::shared_ptr<GraphType> fragment =
      std::dynamic_pointer_cast<GraphType>(client.GetObject(id));
  std::ofstream ofstream;
  std::string output_path = "/Users/weibin/Dev/GraphScope/analytical_engine/build/graph/r" + std::to_string(fragment->fid());
  ofstream.open(output_path);
  for (auto v : fragment->InnerVertices(0)) {
    for (auto e : fragment->GetOutgoingAdjList(v, 0)) {
      ofstream << fragment->GetId(v) << "," << fragment->GetId(e.neighbor()) << "\n";
    }
  }
  ofstream.close();
  // std::shared_ptr<FragmentType> projected_fragment =
  //     FragmentType::Project(fragment, 0, 0, 0, 0);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf(
        "usage: ./run_vy_ldbc <ipc_socket> <e_label_num> <efiles...> "
        "<v_label_num> <vfiles...> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  std::string graph_yaml_path = std::string(argv[index++]);

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    vineyard::Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));

    LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

    vineyard::ObjectID fragment_id;
    {
      std::string relative_location = "/Users/weibin/Dev/gsf/test/yaml_example";
      auto graph_info = gsf::GraphInfo::Make(graph_yaml_path, relative_location);
      auto builder = std::make_unique<
          vineyard::ArrowFragmentBuilder<int64_t,
                                   vineyard::property_graph_types::VID_TYPE>>(
              client, comm_spec, graph_info);
      fragment_id =
          bl::try_handle_all([&builder]() { return builder->LoadFragment(); },
                             [](const vineyard::GSError& e) {
                               LOG(FATAL) << e.error_msg;
                               return 0;
                             },
                             [](const bl::error_info& unmatched) {
                               LOG(FATAL) << "Unmatched error " << unmatched;
                               return 0;
                             });
    }

    LOG(INFO) << "[worker-" << comm_spec.worker_id()
              << "] loaded graph to vineyard ...";

    MPI_Barrier(comm_spec.comm());

    Run(client, comm_spec, fragment_id);

    MPI_Barrier(comm_spec.comm());
  }

  grape::FinalizeMPIComm();
  return 0;
}
