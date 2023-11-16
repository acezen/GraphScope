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
#include "boost/leaf/error.hpp"

#include "apps/pagerank/pagerank_gart.h"
#include "apps/sssp/sssp_gart.h"
#include "apps/wcc/wcc_gart.h"

#include "core/grin/fragment/grin_projected_fragment.h"
// #include "core/fragment/arrow_flattened_fragment.h"

namespace bl = boost::leaf;

using GRINProjectedFragmentType =
    gs::GRINProjectedFragment<int64_t, uint64_t, double,
                              int64_t>;

template <typename FRAG_T>
std::shared_ptr<FRAG_T> GetFragment(const grape::CommSpec& comm_spec, std::string& uri) {
  LOG(FATAL) << "Unimpl";
}

template<>
std::shared_ptr<GRINProjectedFragmentType> GetFragment(const grape::CommSpec& comm_spec, std::string& uri) {
  LOG(INFO) << "Load as GRIN fragment with uri: " << uri;

  GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(uri.c_str());
  LOG(INFO) << comm_spec.fid() << " 1";
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  LOG(INFO) << comm_spec.fid() << " 2";

  size_t local_pnum = grin_get_partition_list_size(pg, local_partitions);
  LOG(INFO) << comm_spec.fid() << " 3 " << local_pnum;
  GRIN_PARTITION partition;
  if (local_pnum == 1) {
    partition = grin_get_partition_from_list(pg, local_partitions, 0);
  } else {
    partition = grin_get_partition_from_list(pg, local_partitions, comm_spec.fid() % local_pnum);
  }
  LOG(INFO) << comm_spec.fid() << " 4";

  return std::make_shared<GRINProjectedFragmentType>(pg, partition, "user", "dist", "knows", "weight");
}

void Run(std::shared_ptr<GRINProjectedFragmentType> frag,
         const grape::CommSpec& comm_spec,
         const std::string& out_prefix,
         std::string& output_result) {
  LOG(INFO) << "Inner vertex number: " << frag->GetInnerVerticesNum();
  auto inner_vertices = frag->InnerVertices();

  std::ofstream file_out;
  file_out.open("traverse/frag-" + std::to_string(comm_spec.fid()) + ".txt");

  auto iter = inner_vertices.begin();
  while (!iter.is_end()) {
    auto v = *iter;
    // LOG(INFO) << "Vertex: " << frag->GetId(v);
    // file_out << frag->GetId(v) << std::endl;
    // auto out_edges = frag->GetOutgoingAdjList(v);
    auto out_edges = frag->GetIncomingAdjList(v);
    auto e_iter = out_edges.begin();
    while (!e_iter.is_end()) {
      auto neighbor = e_iter.neighbor();
      file_out << frag->GetId(v) << " " << frag->GetId(neighbor) << " " << static_cast<double>(e_iter.get_data()) << std::endl;
      // file_out << frag->GetId(v) << " " << frag->GetId(neighbor) << " " << static_cast<double>(e_iter.get_data()) << std::endl;
      ++e_iter;
    }
    ++iter;
  }
  file_out.close();

  // auto fg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
  //     client.GetObject(fragment_group_id));
  // auto fid = comm_spec.WorkerToFrag(comm_spec.worker_id());
  // auto frag_id = fg->Fragments().at(fid);
  // auto arrow_frag = std::static_pointer_cast<FragmentType>(client.GetObject(frag_id));
  // auto frag = std::make_shared<FlattenFragmentType>(arrow_frag.get(), 0, 0);
}

template <typename FRAG_T>
void RunProjectedPR(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix,
                    std::string& output_result) {
  using AppType = gs::PageRankGart<FRAG_T>;
  LOG(INFO) << "Start to create app.";
  auto app = std::make_shared<AppType>();
  LOG(INFO) << "Start to create worker.";
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  LOG(INFO) << "Start to init worker.";
  worker->Init(comm_spec, spec);

  LOG(INFO) << "Start query.";
  double start = grape::GetCurrentTime();

  worker->Query(0.85, 10, 1e-9);
  MPI_Barrier(comm_spec.comm());

  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << "seconds";
  }

  LOG(INFO) << "End query. fid: " << fragment->fid();

  if (output_result == "true") {
    std::ofstream ostream;
    std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
    
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
  }

  worker->Finalize();
}

template <typename FRAG_T>
void RunSSSP(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix,
                    std::string& output_result) {
  using AppType = gs::SSSPGart<FRAG_T>;
  LOG(INFO) << "Start to create app.";
  auto app = std::make_shared<AppType>();
  LOG(INFO) << "Start to create worker.";
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  LOG(INFO) << "Start to init worker.";
  worker->Init(comm_spec, spec);

  LOG(INFO) << "Start query.";
  double start = grape::GetCurrentTime();

  worker->Query(6);

  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << "seconds";
  }

  LOG(INFO) << "End query. fid: " << fragment->fid();

  if (output_result == "true") {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());

    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
  }

  worker->Finalize();
}

template <typename FRAG_T>
void RunWCC(std::shared_ptr<FRAG_T> fragment,
            const grape::CommSpec& comm_spec,
            const std::string& out_prefix,
            std::string& output_result) {
  using AppType = gs::WCCGart<FRAG_T>;
  LOG(INFO) << "Start to create app.";
  auto app = std::make_shared<AppType>();
  LOG(INFO) << "Start to create worker.";
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  LOG(INFO) << "Start to init worker.";
  worker->Init(comm_spec, spec);

  LOG(INFO) << "Start query.";
  double start = grape::GetCurrentTime();

  worker->Query();

  if (comm_spec.worker_id() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << "seconds";
  }

  LOG(INFO) << "End query. fid: " << fragment->fid();

  if (output_result == "true") {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());

    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
  }

  worker->Finalize();
}


void RunGrin(const grape::CommSpec& comm_spec, int argc, char** argv) {
  int index = 2;

  std::string uri = std::string(argv[index++]);
  std::string app_name = std::string(argv[index++]);
  std::string output_result = std::string(argv[index++]);

  uri = "gart://127.0.0.1:23760?read_epoch=0&total_partition_num=4&local_partition_num=1&start_partition_id=" + std::to_string(comm_spec.fid()) + "&meta_prefix=gart_meta_";

  auto frag = GetFragment<GRINProjectedFragmentType>(comm_spec, uri);

  LOG(INFO) << "GetFragment end ....";

  if (app_name == "pagerank") {
    RunProjectedPR<GRINProjectedFragmentType>(frag, comm_spec, "/tmp/output_pr", output_result);
  } else if (app_name == "sssp") {
    RunSSSP<GRINProjectedFragmentType>(frag, comm_spec, "/tmp/output_sssp", output_result);
  } else if (app_name == "wcc") {
    RunWCC<GRINProjectedFragmentType>(frag, comm_spec, "/tmp/output_wcc", output_result);
  } else if (app_name == "traverse") {
    Run(frag, comm_spec, "/tmp/traverse", output_result);
  } else {
    LOG(FATAL) << "Unknown app name: " << app_name;
  }

  LOG(INFO) << "RunGrin end ...";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf(
        "usage: ./run_gart_app <cmd_type> ...\n");
    return 1;
  }
  int index = 1;
  std::string cmd_type = std::string(argv[index++]);

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    if (cmd_type == "run_grin") {
      RunGrin(comm_spec, argc, argv);
    }

    MPI_Barrier(comm_spec.comm());
  }

  LOG(INFO) << "Before FinalizeMPIComm...";

  grape::FinalizeMPIComm();
  return 0;
}

template class gs::GRINProjectedFragment<int64_t, uint64_t, double,
                               int64_t>;
// template class gs::GRINFlattenedFragment<std::string, uint64_t, grape::EmptyType,
//                                           grape::EmptyType>;
