export OMP_NUM_THREADS=16
export OMP_WAIT_POLICY=ACTIVE
export MLPERF_DTYPE=int8
export MLPERF_BATCH=2
sh ./run_batch.sh
