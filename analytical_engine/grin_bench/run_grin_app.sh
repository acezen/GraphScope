# !/bin/bash
set -e pipefail

np=1
cmd_prefix="mpirun"
if ompi_info; then
  echo "Using openmpi"
  cmd_prefix="env LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64  ${cmd_prefix} --allow-run-as-root -x LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 "
fi

socket_file=/tmp/vineyard-gae.sock
executable=/workspaces/GraphScope-GRIN/analytical_engine/build/run_grin_app

cmd="${cmd_prefix} -n ${np} ${executable} run_grin 'v6d://43205470959231526?ipc_socket=${socket_file}' pagerank"

echo "${cmd}"
eval "${cmd}"
echo "Finished running benchmark on property graph."