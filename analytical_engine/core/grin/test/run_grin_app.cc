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

#include "core/grin/fragment/grin_projected_fragment.h"

#include "apps/pagerank/pagerank_networkx.h"
#include "apps/centrality/eigenvector/eigenvector_centrality.h"

#include "apps/projected/sssp_projected.h"
#include "apps/projected/wcc_projected.h"

#include "core/fragment/arrow_flattened_fragment.h"
#include "core/fragment/arrow_projected_fragment.h"
#include "core/loader/arrow_fragment_loader.h"

namespace bl = boost::leaf;

using oid_t = vineyard::property_graph_types::OID_TYPE;
using vid_t = vineyard::property_graph_types::VID_TYPE;

using FragmentType = vineyard::ArrowFragment<oid_t, vid_t>;

using GRINProjectedFragmentType =
    gs::GRINProjectedFragment<oid_t, vid_t, double,
                               int64_t>;

using ProjectedFragmentType =
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

  worker->Query(0.85, 10, 1e-9);
  MPI_Barrier(comm_spec.comm());
  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << "seconds";
  }


  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();

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

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();

  worker->Finalize();
}

template <typename FRAG_T>
void RunWCC(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix) {
  // using AppType = grape::SSSP<FRAG_T>;
  using AppType = gs::WCCProjected<FRAG_T>;
  auto app = std::make_shared<AppType>();
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  worker->Init(comm_spec, spec);

  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Start query.";
  }
  double start = grape::GetCurrentTime();
  worker->Query();
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

std::shared_ptr<GRINProjectedFragmentType> GetGrinFragment(
    const grape::CommSpec& comm_spec,const std::string& uri) {
  LOG(INFO) << "Load as GRIN Fragment";
  GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(uri.c_str());
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  size_t local_pnum = grin_get_partition_list_size(pg, local_partitions);
  GRIN_PARTITION partition;
  if (local_pnum == 1) {
    partition = grin_get_partition_from_list(pg, local_partitions, 0);
  } else {
    partition = grin_get_partition_from_list(pg, local_partitions, comm_spec.fid() % local_pnum);
  }

  return std::make_shared<GRINProjectedFragmentType>(pg, partition, "person", "dist", "knows", "weight");
}

std::shared_ptr<ProjectedFragmentType> GetFragment(const std::string& ipc_socket, vineyard::Client& client,
         const grape::CommSpec& comm_spec,
         vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Load as ARROW Fragment";
  auto fg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
      client.GetObject(fragment_group_id));
  auto fid = comm_spec.WorkerToFrag(comm_spec.worker_id());
  auto frag_id = fg->Fragments().at(fid);
  auto arrow_frag = std::static_pointer_cast<FragmentType>(client.GetObject(frag_id));
  return ProjectedFragmentType::Project(arrow_frag, 0, 0, 0, 0);
}


void RunGrin(const grape::CommSpec& comm_spec, int argc, char** argv) {
  int index = 2;
  std::string uri = std::string(argv[index++]);
  std::string app_name = std::string(argv[index++]);
  auto frag = GetGrinFragment(comm_spec, uri);

  if (app_name == "pagerank") {
    RunProjectedPR<GRINProjectedFragmentType>(frag, comm_spec, "/tmp/output_pr");
  } else if (app_name == "eigenvector") {
    RunProjectedEigen<GRINProjectedFragmentType>(frag, comm_spec, "output_eigen");
  } else if (app_name == "sssp") {
    RunSSSP<GRINProjectedFragmentType>(frag, comm_spec, "/tmp/output_sssp");
  } else if (app_name == "wcc") {
    RunWCC<GRINProjectedFragmentType>(frag, comm_spec, "/tmp/output_wcc");
  } else {
    LOG(FATAL) << "Unknown app name: " << app_name;
  }
  LOG(INFO) << "finish running application ... memory = "
            << vineyard::get_rss_pretty()
            << ", peak = " << vineyard::get_peak_rss_pretty();
}

void RunNoGrin(const grape::CommSpec& comm_spec, int argc, char** argv) {
  int index = 2;
  std::string ipc_socket = std::string(argv[index++]);
  std::string app_name = std::string(argv[index++]);
  vineyard::ObjectID fragment_id = vineyard::InvalidObjectID();
  fragment_id = atol(argv[index++]);

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  auto frag = GetFragment(ipc_socket, client, comm_spec, fragment_id);

  if (app_name == "pagerank") {
    RunProjectedPR<ProjectedFragmentType>(frag, comm_spec, "/tmp/output_pr");
  } else if (app_name == "eigenvector") {
    RunProjectedEigen<ProjectedFragmentType>(frag, comm_spec, "output_eigen");
  } else if (app_name == "sssp") {
    RunSSSP<ProjectedFragmentType>(frag, comm_spec, "/tmp/output_sssp");
  } else if (app_name == "wcc") {
    RunWCC<ProjectedFragmentType>(frag, comm_spec, "/tmp/output_wcc"); 
  } else {
    LOG(FATAL) << "Unknown app name: " << app_name;
  }
  LOG(INFO) << "finish running application ... memory = "
            << vineyard::get_rss_pretty()
            << ", peak = " << vineyard::get_peak_rss_pretty();
}

void LoadGraphToVineyard(int argc, char** argv) {
  LOG(INFO) << "Loading Graph To Vineyard.";
  if (argc < 6) {
    printf(
        "usage: ./run_grin_app <cmd_type> <ipc_socket> <e_label_num> <efiles...> "
        "<v_label_num> <vfiles...> [directed]\n");
    return;
  }
  int index = 2;
  std::string ipc_socket = std::string(argv[index++]);

  int edge_label_num = atoi(argv[index++]);
  std::vector<std::string> efiles;
  for (int i = 0; i < edge_label_num; ++i) {
    efiles.push_back(argv[index++]);
    LOG(INFO) << "efile: " << efiles.back();
  }

  int vertex_label_num = atoi(argv[index++]);
  std::vector<std::string> vfiles;
  for (int i = 0; i < vertex_label_num; ++i) {
    vfiles.push_back(argv[index++]);
    LOG(INFO) << "vfile: " << vfiles.back();
  }

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index++]);
  }

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    vineyard::Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));

    LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
    vineyard::ObjectID fragment_id = vineyard::InvalidObjectID();

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

    MPI_Barrier(comm_spec.comm());
  }
}

void RunApp(int argc, char** argv) {
  int index = 1;
  std::string cmd_type = argv[index++];

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    if (cmd_type == "run_grin") {
      RunGrin(comm_spec, argc, argv);
    } else if (cmd_type == "run_arrow") {
      RunNoGrin(comm_spec, argc, argv);
    }

    MPI_Barrier(comm_spec.comm());
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf(
        "usage: ./run_grin_app <cmd_type> ...\n");
    return 1;
  }
  int index = 1;
  std::string cmd_type = std::string(argv[index++]);
  grape::InitMPIComm();
  if (cmd_type == "load_graph") {
    LoadGraphToVineyard(argc, argv);
  } else {
    RunApp(argc, argv);
  }
  grape::FinalizeMPIComm();
  return 0;
}

template class gs::GRINProjectedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
template class gs::ArrowProjectedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
template class gs::GRINProjectedFragment<int64_t, uint64_t, double,
                               int64_t>;
template class gs::ArrowProjectedFragment<int64_t, uint64_t, double,
                               int64_t>;
