export MLPERF_DTYPE=int8

export MLPERF_BATCH=3
./run_batch.sh

export MLPERF_BATCH=4
./run_batch.sh

export MLPERF_BATCH=5
./run_batch.sh

export MLPERF_BATCH=6
./run_batch.sh

export MLPERF_BATCH=7
./run_batch.sh

export MLPERF_BATCH=8
./run_batch.sh
