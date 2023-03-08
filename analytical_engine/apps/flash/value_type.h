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

#ifndef ANALYTICAL_ENGINE_APPS_FLASH_VALUE_TYPE_H_
#define ANALYTICAL_ENGINE_APPS_FLASH_VALUE_TYPE_H_

#include <vector>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace gs {

struct EMPTY_TYPE { };

struct BFS_TYPE {
  int dis;
};

struct SSSP_TYPE {
  float dis;
};

struct BC_TYPE {
  char d;
  float b, c;
};

struct KATZ_TYPE {
  float val, next;
};

struct CLOSENESS_TYPE {
  int64_t seen, cnt;
  double val;
};

struct HARMONIC_TYPE {
  int64_t seen;
  double val;
};

struct CC_TYPE {
  int res;
};

struct CC_OPT_TYPE {
  int64_t res;
};

struct CC_LOG_TYPE {
  int res, s, f;
};

struct SCC_TYPE {
  int fid, scc;
};

struct BCC_TYPE {
  int d, cid, p, dis, res;
};

struct PR_TYPE {
  int deg;
  double res, next;
};

struct HITS_TYPE {
  float auth, hub, auth1, hub1;
};

struct MIS_2_TYPE {
  bool d, b;
};

struct MM_TYPE {
  int32_t p, s;
};

struct K_CORE_TYPE {
  int32_t d;
};

struct CORE_TYPE {
  short core;
  int cnt;
  std::vector<short> s;
};

struct MIS_TYPE {
  bool d, b;
  int64_t r;
};

struct CORE_2_TYPE {
  short core, old;
};

struct COLOR_TYPE {
  short c, cc;
  int32_t deg;
  std::vector<int> colors;
};

struct K_CLIQUE_2_TYPE {
  int32_t deg, count;
  std::vector<int32_t> out;
};

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const CLOSENESS_TYPE& v) {
  in_archive << v.seen;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, CLOSENESS_TYPE& v) {
  out_archive >> v.seen;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const HARMONIC_TYPE& v) {
  in_archive << v.seen;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, HARMONIC_TYPE& v) {
  out_archive >> v.seen;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const PR_TYPE& v) {
  in_archive << v.deg << v.res;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, PR_TYPE& v) {
  out_archive >> v.deg >> v.res;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const HITS_TYPE& v) {
  in_archive << v.auth << v.hub;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, HITS_TYPE& v) {
  out_archive >> v.auth >> v.hub;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const MIS_TYPE& v) {
  in_archive << v.d << v.r;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, MIS_TYPE& v) {
  out_archive >> v.d >> v.r;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const CORE_TYPE& v) {
  in_archive << v.core;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, CORE_TYPE& v) {
  out_archive >> v.core;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const CORE_2_TYPE& v) {
  in_archive << v.core;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, CORE_2_TYPE& v) {
  out_archive >> v.core;
  return out_archive;
}

struct AB_CORE_TYPE {
  int d, c;
};

struct TRIANGLE_TYPE {
  int32_t deg, count;
  std::set<int32_t> out;
};

struct RECTANGLE_TYPE {
  int32_t deg, count;
  std::vector<std::pair<int, int> > out;
};

struct EGO_TYPE {
  int deg;
  std::vector<int> out;
  std::vector<std::vector<int> > ego;
};

struct LPA_TYPE {
  int c, cc;
  std::vector<int> s;
};

struct MATRIX_TYPE {
  std::vector<float> val;
};

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const TRIANGLE_TYPE& v) {
  in_archive << v.deg << v.out;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, TRIANGLE_TYPE& v) {
  out_archive >> v.deg >> v.out;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const RECTANGLE_TYPE& v) {
  in_archive << v.deg << v.out;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, RECTANGLE_TYPE& v) {
  out_archive >> v.deg >> v.out;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const K_CLIQUE_2_TYPE& v) {
  in_archive << v.deg << v.out;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, K_CLIQUE_2_TYPE& v) {
  out_archive >> v.deg >> v.out;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const MATRIX_TYPE& v) {
  in_archive << v.val;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, MATRIX_TYPE& v) {
  out_archive >> v.val;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const COLOR_TYPE& v) {
  in_archive << v.c << v.deg;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, COLOR_TYPE& v) {
  out_archive >> v.c >> v.deg;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const LPA_TYPE& v) {
  in_archive << v.c;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, LPA_TYPE& v) {
  out_archive >> v.c;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const EGO_TYPE& v) {
  in_archive << v.deg << v.out;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, EGO_TYPE& v) {
  out_archive >> v.deg >> v.out;
  return out_archive;
}

inline grape::InArchive& operator<<(grape::InArchive& in_archive, const KATZ_TYPE& v) {
  in_archive << v.val;
  return in_archive;
}
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive, KATZ_TYPE& v) {
  out_archive >> v.val;
  return out_archive;
}

}  // namespace gs

#endif // ANALYTICAL_ENGINE_APPS_FLASH_VALUE_TYPE_H_
