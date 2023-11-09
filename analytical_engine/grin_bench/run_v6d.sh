#!/bin/bash
set -eo pipefail
export VINEYARD_HOME=/opt/graphscope/bin/

socket_file=/tmp/vineyard-gae.sock
function start_vineyard() {
  # pkill vineyardd || true
  # pkill etcd || true
  echo "[INFO] vineyardd will using the socket_file on ${socket_file}"

  vineyardd \
    --socket ${socket_file} \
    --size "200G" \
    --etcd_prefix "grin-etcd" \
    --etcd_endpoint=localhost:3460 &
  set +m
  sleep 5
  echo "vineyardd started."
}

start_vineyard
