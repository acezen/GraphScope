# !/bin/bash
set -e pipefail

np=1
cmd_prefix="mpirun"
if ompi_info; then
  echo "Using openmpi"
  cmd_prefix="env LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64  ${cmd_prefix} --allow-run-as-root -x LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 "
fi

socket_file=/tmp/vineyard-gae.sock
vfile=/workspaces/GraphScope-GRIN/analytical_engine/grin_bench/gstest/p2p-31.v
efile=/workspaces/GraphScope-GRIN/analytical_engine/grin_bench/gstest/p2p-31.e
executable=/workspaces/GraphScope-GRIN/analytical_engine/build/run_grin_app

cmd="${cmd_prefix} -n ${np} ${executable} load_graph '${socket_file}' 1 '${efile}#src_label=person&dst_label=person&label=knows#header_row=false#delimiter= ' 1 '${vfile}#label=person#header_row=false#delimiter= ' 1"

echo "${cmd}"
eval "${cmd}"
echo "Finished running benchmark on property graph."
