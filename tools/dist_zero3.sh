#!/usr/bin/env bash

set -x

FILE=$1
CONFIG=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((18500 + $RANDOM % 2000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
DEEPSPEED=${DEEPSPEED:-deepspeed_zero3}

if [[ $FILE == *.py ]]; then
    FILE=${FILE}
else
    FILE=tools/${FILE}.py
fi

if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=${NNODES} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    ${FILE} ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    ${FILE} ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
fi
