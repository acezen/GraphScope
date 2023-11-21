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
#include "core/grin/fragment/grin_projected_fragment.h"

#include "apps/pagerank/pagerank_networkx.h"


using oid_t = int64_t;
using vid_t = uint64_t;

using GRINProjectedFragmentType =
    gs::GRINProjectedFragment<oid_t, vid_t, double,
                               int64_t>;

template <typename FRAG_T>
void RunProjectedPR(std::shared_ptr<FRAG_T> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix, std::string& output_result) {
  using AppType = gs::PageRankGraphAr<FRAG_T>;
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

std::shared_ptr<GRINGraphArFragmentType> GetGrinFragment(
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

  return std::make_shared<GRINGraphArFragmentType>(pg, partition, "person", "dist", "knows", "weight");
}


void RunGrin(const grape::CommSpec& comm_spec, int argc, char** argv) {
  int index = 1;
  std::string uri = std::string(argv[index++]);
  std::string app_name = std::string(argv[index++]);
  std::string output_result = std::string(argv[index++]);
  auto frag = GetGrinFragment(comm_spec, uri);

  RunProjectedPR<GRINGraphArFragmentType>(frag, comm_spec, "/tmp/output_pr", output_result);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf(
        "usage: ./run_grin_app <cmd_type> ...\n");
    return 1;
  }
  grape::InitMPIComm();
  RunGrin(argc, argv);
  grape::FinalizeMPIComm();
  return 0;
}

template class gs::GRINProjectedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
template class gs::GRINProjectedFragment<int64_t, uint64_t, double,
                               int64_t>;
