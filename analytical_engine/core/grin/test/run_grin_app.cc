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
// #include "grape/analytical_apps/sssp/sssp.h"
#include "vineyard/client/client.h"
#include "vineyard/graph/fragment/arrow_fragment.h"

#include "core/grin/fragment/arrow_flattened_fragment.grin.h"

#include "apps/pagerank/pagerank_networkx.h"
#include "apps/centrality/eigenvector/eigenvector_centrality.h"

#include "apps/projected/sssp_projected.h"

#include "core/fragment/arrow_flattened_fragment.h"
#include "core/fragment/arrow_projected_fragment.h"
#include "core/loader/arrow_fragment_loader.h"

namespace bl = boost::leaf;

using oid_t = vineyard::property_graph_types::OID_TYPE;
// using oid_t = std::string;
using vid_t = vineyard::property_graph_types::VID_TYPE;

using FragmentType = vineyard::ArrowFragment<oid_t, vid_t>;

using GRINFlattenFragmentType =
    gs::GRINFlattenedFragment<oid_t, vid_t, double,
                               int64_t>;
// using FlattenFragmentType =
//     gs::ArrowFlattenedFragment<oid_t, vid_t, int64_t,
//                                int64_t>;

using FlattenFragmentType =
    gs::ArrowProjectedFragment<oid_t, vid_t, double,
                               int64_t>;
template <typename FRAG_T>
void RunProjectedPR(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix) {
  using AppType = gs::PageRankNetworkX<FRAG_T>;
  auto app = std::make_shared<AppType>();
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Start Init worker.";
  }
  worker->Init(comm_spec, spec);

  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Start query.";
  }
  double start = grape::GetCurrentTime();

  worker->Query(0.85, 10, 1e-6);
  MPI_Barrier(comm_spec.comm());
  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << "seconds";
  }


  /*
  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  */

  worker->Finalize();
}

template <typename FRAG_T>
void RunSSSP(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix) {
  // using AppType = grape::SSSP<FRAG_T>;
  using AppType = gs::SSSPProjected<FRAG_T>;
  auto app = std::make_shared<AppType>();
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  worker->Init(comm_spec, spec);

  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Start query.";
  }
  double start = grape::GetCurrentTime();
  worker->Query(6);
  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << "seconds";
  }

  /*
  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  */

  worker->Finalize();
}

template <typename FRAG_T>
void RunProjectedEigen(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix) {
  // using AppType = grape::PageRankAuto<ProjectedFragmentType>;
  using AppType = gs::EigenvectorCentrality<FRAG_T>;
  // using AppType = grape::PageRankLocal<ProjectedFragmentType>;
  auto app = std::make_shared<AppType>();
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  worker->Init(comm_spec, spec);

  worker->Query(1e-6, 10);

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();

  worker->Finalize();
}

template <typename FRAG_T>
std::shared_ptr<FRAG_T> GetFragment(const std::string& ipc_socket, vineyard::Client& client,
         const grape::CommSpec& comm_spec,
         vineyard::ObjectID fragment_group_id) {
  LOG(FATAL) << "Unimpl";
}

template<>
std::shared_ptr<GRINFlattenFragmentType> GetFragment(const std::string& ipc_socket, vineyard::Client& client,
         const grape::CommSpec& comm_spec,
         vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Load As GRIN Fragment";
  std::string fg_id_str = std::to_string(fragment_group_id);

  char** argv = new char*[2];
  argv[0] = new char[ipc_socket.length() + 1];
  argv[1] = new char[fg_id_str.length() + 1];

  strcpy(argv[0], ipc_socket.c_str());
  strcpy(argv[1], fg_id_str.c_str());

  GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(2, argv);
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  size_t local_pnum = grin_get_partition_list_size(pg, local_partitions);
  GRIN_PARTITION partition;
  if (local_pnum == 1) {
    partition = grin_get_partition_from_list(pg, local_partitions, 0);
  } else {
    partition = grin_get_partition_from_list(pg, local_partitions, comm_spec.fid() % local_pnum);
  }

  return std::make_shared<GRINFlattenFragmentType>(pg, partition, "dist", "weight");
}

template<>
std::shared_ptr<FlattenFragmentType> GetFragment(const std::string& ipc_socket, vineyard::Client& client,
         const grape::CommSpec& comm_spec,
         vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Load As ARROW Fragment";
  auto fg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
      client.GetObject(fragment_group_id));
  auto fid = comm_spec.WorkerToFrag(comm_spec.worker_id());
  auto frag_id = fg->Fragments().at(fid);
  auto arrow_frag = std::static_pointer_cast<FragmentType>(client.GetObject(frag_id));
  //  return std::make_shared<FlattenFragmentType>(arrow_frag.get(), 0, 0);
  return FlattenFragmentType::Project(arrow_frag, 0, 0, 0, 0);
}

template<typename FRAG_T>
void Run(const std::string& ipc_socket, vineyard::Client& client,
         const grape::CommSpec& comm_spec,
         vineyard::ObjectID fragment_group_id, const std::string& app_name) {
  /*
  std::string fg_id_str = std::to_string(fragment_group_id);

  char** argv = new char*[2];
  argv[0] = new char[ipc_socket.length() + 1];
  argv[1] = new char[fg_id_str.length() + 1];

  strcpy(argv[0], ipc_socket.c_str());
  strcpy(argv[1], fg_id_str.c_str());

  GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(2, argv);
  auto total_pnum = grin_get_total_partitions_number(pg);
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  size_t local_pnum = grin_get_partition_list_size(pg, local_partitions);
  GRIN_PARTITION partition;
  if (local_pnum == 1) {
    partition = grin_get_partition_from_list(pg, local_partitions, 0);
  } else {
    CHECK(local_pnum == total_pnum);
    partition = grin_get_partition_from_list(pg, local_partitions, comm_spec.fid());
  }

  auto frag = std::make_shared<FlattenFragmentType>(
    pg, partition, "weight", "dist");
  */
  auto frag = GetFragment<FRAG_T>(ipc_socket, client, comm_spec, fragment_group_id);

  if (app_name == "pagerank") {
    RunProjectedPR<FRAG_T>(frag, comm_spec, "/tmp/output_pr2");
  } else if (app_name == "eigenvector") {
    RunProjectedEigen<FRAG_T>(frag, comm_spec, "output_eigen");
  } else if (app_name == "sssp") {
    RunSSSP<FRAG_T>(frag, comm_spec, "/tmp/output_sssp");
  } else {
    LOG(FATAL) << "Unknown app name: " << app_name;
  }
  LOG(INFO) << "finish running application ... memory = "
            << vineyard::get_rss_pretty()
            << ", peak = " << vineyard::get_peak_rss_pretty();
}

int main(int argc, char** argv) {
  if (argc < 6) {
    printf(
        "usage: ./run_vy_app <ipc_socket> <e_label_num> <efiles...> "
        "<v_label_num> <vfiles...> <run_projected>"
        "[directed] [app_name] [path_pattern]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int edge_label_num = atoi(argv[index++]);
  std::vector<std::string> efiles;
  for (int i = 0; i < edge_label_num; ++i) {
    efiles.push_back(argv[index++]);
  }

  int vertex_label_num = atoi(argv[index++]);
  std::vector<std::string> vfiles;
  for (int i = 0; i < vertex_label_num; ++i) {
    vfiles.push_back(argv[index++]);
  }

  int directed = 1;
  std::string frag_type = "";
  std::string app_name = "";
  if (argc > index) {
    directed = atoi(argv[index++]);
  }
  if (argc > index) {
    frag_type = argv[index++];
  }
  if (argc > index) {
    app_name = argv[index++];
  }

  vineyard::ObjectID fragment_id = vineyard::InvalidObjectID();
  if (argc > index) {
    fragment_id = atol(argv[index++]);
  }
  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    vineyard::Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));

    LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
    if (fragment_id == vineyard::InvalidObjectID())
    {
      auto loader = std::make_unique<gs::ArrowFragmentLoader<oid_t, vid_t>>(
          client, comm_spec, efiles, vfiles, directed != 0,
          /* generate_eid */ false, /* retain_oid */ false);
      fragment_id =
          bl::try_handle_all([&loader]() { return loader->LoadFragmentAsFragmentGroup(); },
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
              << "] loaded graph to vineyard ... " << fragment_id;
    LOG(INFO) << "peek memory: " << vineyard::get_peak_rss_pretty()
              << std::endl;

    MPI_Barrier(comm_spec.comm());

    if (frag_type == "grin") {
      Run<GRINFlattenFragmentType>(ipc_socket, client, comm_spec, fragment_id, app_name);
    } else if (frag_type == "arrow") {
      Run<FlattenFragmentType>(ipc_socket, client, comm_spec, fragment_id, app_name);
    }
    LOG(INFO) << "memory: " << vineyard::get_rss_pretty()
              << ", peek memory: " << vineyard::get_peak_rss_pretty();

    MPI_Barrier(comm_spec.comm());
  }

  grape::FinalizeMPIComm();
  return 0;
}

template class gs::GRINFlattenedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
template class gs::ArrowFlattenedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
template class gs::ArrowProjectedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
template class gs::GRINFlattenedFragment<int64_t, uint64_t, double,
                               int64_t>;
template class gs::ArrowProjectedFragment<int64_t, uint64_t, double,
                               int64_t>;
