#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>     
#include "vineyard/graph/grin/predefine.h"
#include "grin/common/error.h"
#include "grin/common/message.h"
#include "grin/index/external_id.h"
#include "grin/index/internal_id.h"
#include "grin/index/label.h"
#include "grin/index/order.h"
#include "grin/index/pk.h"
#include "grin/partition/partition.h"
#include "grin/partition/reference.h"
#include "grin/partition/topology.h"
#include "grin/property/primarykey.h"
#include "grin/property/property.h"
#include "grin/property/propertylist.h"
#include "grin/property/row.h"
#include "grin/property/topology.h"
#include "grin/property/type.h"
#include "grin/property/value.h"
#include "grin/topology/adjacentlist.h"
#include "grin/topology/edgelist.h"
#include "grin/topology/structure.h"
#include "grin/topology/vertexlist.h"

GRIN_GRAPH get_graph(int argc, char** argv, int p) {
#ifdef GRIN_ENABLE_GRAPH_PARTITION
  GRIN_PARTITIONED_GRAPH pg =
      grin_get_partitioned_graph_from_storage(argv[1]);
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  assert(p < grin_get_partition_list_size(pg, local_partitions));
  GRIN_PARTITION partition =
      grin_get_partition_from_list(pg, local_partitions, p);
  GRIN_PARTITION_ID partition_id = grin_get_partition_id(pg, partition);
  GRIN_PARTITION p1 = grin_get_partition_by_id(pg, partition_id);
  if (!grin_equal_partition(pg, partition, p1)) {
    printf("partition not match\n");
  }
  grin_destroy_partition(pg, p1);
  GRIN_GRAPH g = grin_get_local_graph_by_partition(pg, partition);
  grin_destroy_partition(pg, partition);
  grin_destroy_partition_list(pg, local_partitions);
  grin_destroy_partitioned_graph(pg);
#else
  GRIN_GRAPH g = grin_get_graph_from_storage(argv[1]);
#endif
  return g;
}

int main(int argc, char** argv) {
  GRIN_GRAPH g = get_graph(argc, argv, 0);

  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);

  GRIN_VERTEX_TYPE vt = grin_get_vertex_type_by_name(g, "person");

  long long int lb = grin_get_vertex_internal_id_lower_bound_by_type(g, vt);
  long long int ub = grin_get_vertex_internal_id_upper_bound_by_type(g, vt);

  printf("lb: %lld, ub: %lld\n", lb, ub);

  long long* comp = (long long*)malloc(sizeof(long long) * (ub - lb));
  bool* next_modified = (bool*)malloc(sizeof(bool) * (ub - lb));

  GRIN_VERTEX_LIST iv = grin_get_vertex_list_by_type_select_master(g, vt);
  size_t iv_size = grin_get_vertex_list_size(g, iv);

  for (size_t i = 0; i < iv_size; ++i) {
    GRIN_VERTEX v = grin_get_vertex_from_list(g, iv, i);
    long long int iid = grin_get_vertex_internal_id_by_type(g, vt, v);
    GRIN_VERTEX_REF vf = grin_get_vertex_ref_by_vertex(g, v);
    long long int ivf = grin_serialize_vertex_ref_as_int64(g, vf);

    if (i == 0 || i == iv_size - 1) {
      printf("Inner %lld %lld\n", iid, ivf);
    }

    comp[iid - lb] = ivf;
    grin_destroy_vertex(g, v);
    grin_destroy_vertex_ref(g, vf);
  }

  GRIN_VERTEX_LIST ov = grin_get_vertex_list_by_type_select_mirror(g, vt);
  size_t ov_size = grin_get_vertex_list_size(g, ov);

  for (size_t i = 0; i < ov_size; ++i) {
    GRIN_VERTEX v = grin_get_vertex_from_list(g, ov, i);
    long long int iid = grin_get_vertex_internal_id_by_type(g, vt, v);
    GRIN_VERTEX_REF vf = grin_get_vertex_ref_by_vertex(g, v);
    long long int ivf = grin_serialize_vertex_ref_as_int64(g, vf);

    if (i == 0 || i == ov_size - 1) {
      printf("Outer %lld %lld\n", iid, ivf);
    }

    comp[iid - lb] = ivf;
    grin_destroy_vertex(g, v);
    grin_destroy_vertex_ref(g, vf);
  }


  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0; 
  printf("------------- 2 -------------: %lf s.\n", elapsedTime);

  GRIN_EDGE_TYPE et = grin_get_edge_type_by_name(g, "knows");
  for (size_t i = 0; i < iv_size; ++i) {
    GRIN_VERTEX v = grin_get_vertex_from_list(g, iv, i);
    long long int iid = grin_get_vertex_internal_id_by_type(g, vt, v) - lb;
    long long cid = comp[iid];

    GRIN_ADJACENT_LIST al = grin_get_adjacent_list_by_edge_type(g, OUT, v, et);
    size_t al_size = grin_get_adjacent_list_size(g, al);
    for (size_t j = 0; j < al_size; ++j) {
      GRIN_VERTEX u = grin_get_neighbor_from_adjacent_list(g, al, j);
      long long int uid = grin_get_vertex_internal_id_by_type(g, vt, u) - lb;
      if (comp[uid] > cid) {
        comp[uid] = cid;
        next_modified[uid] = true;
      }
      grin_destroy_vertex(g, u);
    }
    grin_destroy_adjacent_list(g, al);

    al = grin_get_adjacent_list_by_edge_type(g, IN, v, et);
    al_size = grin_get_adjacent_list_size(g, al);
    for (size_t j = 0; j < al_size; ++j) {
      GRIN_VERTEX u = grin_get_neighbor_from_adjacent_list(g, al, j);
      long long int uid = grin_get_vertex_internal_id_by_type(g, vt, u) - lb;
      if (comp[uid] > cid) {
        comp[uid] = cid;
        next_modified[uid] = true;
      }
      grin_destroy_vertex(g, u);
    }
  }

  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0; 
  printf("------------- 3 -------------: %lf s.\n", elapsedTime);

  grin_destroy_vertex_list(g, iv);
  grin_destroy_vertex_list(g, ov);
  grin_destroy_vertex_type(g, vt);
  grin_destroy_graph(g);
  return 0;
}