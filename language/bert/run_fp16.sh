export MLPERF_DTYPE=fp16

cp user_fp16.conf user.conf

export MLPERF_BATCH=1
sh ./run_batch.sh

export MLPERF_BATCH=4
sh ./run_batch.sh

export MLPERF_BATCH=8
sh ./run_batch.sh

export MLPERF_BATCH=16
sh ./run_batch.sh

export MLPERF_BATCH=20
sh ./run_batch.sh

export MLPERF_BATCH=24
sh ./run_batch.sh

export MLPERF_BATCH=28
sh ./run_batch.sh

export MLPERF_BATCH=32
sh ./run_batch.sh
