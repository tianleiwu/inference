export MLPERF_DTYPE=fp16

export MLPERF_BATCH=3
sh ./run_batch.sh

export MLPERF_BATCH=4
sh ./run_batch.sh

export MLPERF_BATCH=5
sh ./run_batch.sh

export MLPERF_BATCH=6
sh ./run_batch.sh

export MLPERF_BATCH=7
sh ./run_batch.sh

export MLPERF_BATCH=8
sh ./run_batch.sh
