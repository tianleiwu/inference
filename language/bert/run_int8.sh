export OMP_NUM_THREADS=16
export OMP_WAIT_POLICY=ACTIVE
export MLPERF_DTYPE=int8

cp user_int8.conf user.conf

export MLPERF_BATCH=1
sh ./run_batch.sh

export MLPERF_BATCH=2
sh ./run_batch.sh

export MLPERF_BATCH=3
sh ./run_batch.sh

export MLPERF_BATCH=4
sh ./run_batch.sh