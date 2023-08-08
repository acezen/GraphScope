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

#include "apps/pagerank/pagerank_gart.h"
#include "apps/sssp/sssp_gart.h"

#include "core/grin/fragment/arrow_flattened_fragment.grin.h"
// #include "core/fragment/arrow_flattened_fragment.h"

namespace bl = boost::leaf;

using FlattenFragmentType =
    gs::GRINFlattenedFragment<int64_t, uint64_t, int64_t,
                              int64_t>;

template <typename FRAG_T>
std::shared_ptr<FRAG_T> GetFragment(char* uri, const grape::CommSpec& comm_spec) {
  LOG(FATAL) << "Unimpl";
}

template<>
std::shared_ptr<FlattenFragmentType> GetFragment(char* uri, const grape::CommSpec& comm_spec) {
  LOG(INFO) << "Load As GRIN Fragment";
  GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(uri);
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  size_t local_pnum = grin_get_partition_list_size(pg, local_partitions);
  GRIN_PARTITION partition;
  if (local_pnum == 1) {
    partition = grin_get_partition_from_list(pg, local_partitions, 0);
  } else {
    partition = grin_get_partition_from_list(pg, local_partitions, comm_spec.fid() % local_pnum);
  }
  return std::make_shared<FlattenFragmentType>(
    pg, partition, "person_id", "weight");
}

void Run(char* uri, const grape::CommSpec& comm_spec) {
  auto frag = GetFragment<FlattenFragmentType>(uri, comm_spec);
  LOG(INFO) << "Inner vertex number: " << frag->GetInnerVerticesNum();
  LOG(INFO) << "Edge number: " << frag->GetEdgeNum();
  auto inner_vertices = frag->InnerVertices();
  auto iter = inner_vertices.begin();
  std::ofstream file_out;
  file_out.open("traverse/frag-" + std::to_string(comm_spec.fid()) + ".txt");
  while (!iter.is_end()) {
    auto v = *iter;
    LOG(INFO) << "Vertex: " << frag->GetId(v);
    auto out_edges = frag->GetOutgoingAdjList(v);
    auto e_iter = out_edges.begin();
    while (!e_iter.is_end()) {
      auto neighbor = e_iter.neighbor();
      file_out << frag->GetId(v) + 1 << " " << frag->GetId(neighbor) + 1  << " " << e_iter.get_data() << std::endl;
      ++e_iter;
    }
    ++iter;
  }
  file_out.close();
  /*
  auto fg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
      client.GetObject(fragment_group_id));
  auto fid = comm_spec.WorkerToFrag(comm_spec.worker_id());
  auto frag_id = fg->Fragments().at(fid);
  auto arrow_frag = std::static_pointer_cast<FragmentType>(client.GetObject(frag_id));
  auto frag = std::make_shared<FlattenFragmentType>(arrow_frag.get(), 0, 0);
  */
}

void RunProjectedPR(std::shared_ptr<FlattenFragmentType> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix) {
  // using AppType = grape::PageRankAuto<ProjectedFragmentType>;
  using AppType = gs::PageRankGart<FlattenFragmentType>;
  // using AppType = grape::PageRankLocal<ProjectedFragmentType>;
  LOG(INFO) << "Start to create app.";
  auto app = std::make_shared<AppType>();
  LOG(INFO) << "Start to create worker.";
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  LOG(INFO) << "Start to init worker.";
  worker->Init(comm_spec, spec);

  LOG(INFO) << "Start query.";
  double start = grape::GetCurrentTime();
  worker->Query(0.85, 5);
  if (fragment->fid() == 0) {
    LOG(INFO) << "Query time: " << grape::GetCurrentTime() - start << " seconds";
  }
  LOG(INFO) << "End query. fid: " << fragment->fid();

  std::ofstream ostream;
  std::string output_path = "/root/wanglei/grin_pr_result_frag_"+std::to_string(fragment->fid());
  //    grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();

  worker->Finalize();
}

void RunProjectedSSSP(std::shared_ptr<FlattenFragmentType> fragment,
                    const grape::CommSpec& comm_spec,
                    const std::string& out_prefix) {
  // using AppType = grape::PageRankAuto<ProjectedFragmentType>;
  using AppType = gs::SSSPGart<FlattenFragmentType>;
  // using AppType = grape::PageRankLocal<ProjectedFragmentType>;
  LOG(INFO) << "Start to create app.";
  auto app = std::make_shared<AppType>();
  LOG(INFO) << "Start to create worker.";
  auto worker = AppType::CreateWorker(app, fragment);
  auto spec = grape::DefaultParallelEngineSpec();
  LOG(INFO) << "Start to init worker.";
  worker->Init(comm_spec, spec);

  LOG(INFO) << "Start query.";
  worker->Query(5);
  LOG(INFO) << "End query. fid: " << fragment->fid();

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());

  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();

  worker->Finalize();
}

void RunPagerank(char* uri, const grape::CommSpec& comm_spec) {
  auto frag = GetFragment<FlattenFragmentType>(uri, comm_spec);
  LOG(INFO) << "Start to run pagerank.";
  RunProjectedPR(frag, comm_spec,  "./output_projected_pagerank/");
  // RunProjectedSSSP(frag, comm_spec,  "./output_projected_pagerank/");
}

int main(int argc, char** argv) {
  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    std::string uri_str = "gart://192.168.0.22:23760?read_epoch=1&total_partition_num=4&local_partition_num=1&start_partition_id="+std::to_string(comm_spec.fid())+"&meta_prefix=gart_meta_";

    char* uri = const_cast<char*>(uri_str.c_str());

    RunPagerank(uri, comm_spec);

    /*
    int read_epoch = 1;
    std::string etcd_endpoint = "http://192.168.0.22:23760";
    std::string meta_prefix = "gart_meta_";
    char** argv = new char*[5];
    argv[0] = new char[etcd_endpoint.length() + 1];
    argv[1] = new char[std::to_string(comm_spec.fnum()).length() + 1];
    argv[2] = new char[std::to_string(comm_spec.fid()).length() + 1];
    argv[3] = new char[std::to_string(read_epoch).length() + 1];
    argv[4] = new char[meta_prefix.length() + 1];

    strcpy(argv[0], etcd_endpoint.c_str());
    strcpy(argv[1], std::to_string(comm_spec.fnum()).c_str());
    strcpy(argv[2], std::to_string(comm_spec.fid()).c_str());
    strcpy(argv[3], std::to_string(read_epoch).c_str());
    strcpy(argv[4], meta_prefix.c_str());
    RunPagerank(argv, comm_spec);
    */
    // Run(argv, comm_spec);
  }

  grape::FinalizeMPIComm();
  return 0;
}

template class gs::GRINFlattenedFragment<int64_t, uint64_t, int64_t,
                               int64_t>;
// template class gs::GRINFlattenedFragment<std::string, uint64_t, grape::EmptyType,
//                                           grape::EmptyType>;
