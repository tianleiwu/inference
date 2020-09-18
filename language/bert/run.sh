# Run MLPerf on Azure VM (prefer E*s_v4 with VNNI for int8, or T4 for fp16)
# Test boxes: e32 has 16 physical cores, e16 has 8 physical cores, and t4 is T4 GPU
HELP="Usage: sh run.sh offline_batch_size run_id e32|e16|t4 [num_threads]"
if [ -z "$1" ]; then
  echo "${HELP}"
  exit 1
fi

if [ -z "$2" ]; then
  echo "${HELP}"
  exit 1
fi

NUM_THREADS=0
if [ "$3" = "e32" ]; then
    export OMP_NUM_THREADS=${4:-14}
    export OMP_WAIT_POLICY=ACTIVE
    QTYPE=int8
    TEST_BOX=E32
    cp user_int8.conf user.conf
    NUM_THREADS=${OMP_NUM_THREADS}
elif [ "$3" = "e16" ]; then
    export OMP_NUM_THREADS=${4:-7}
    export OMP_WAIT_POLICY=ACTIVE
    QTYPE=int8
    TEST_BOX=E16
    cp user_int8.conf user.conf
    NUM_THREADS=${OMP_NUM_THREADS}
elif [ "$3" = "t4" ]; then
    QTYPE=fp16
    TEST_BOX=T4
    cp user_fp16.conf user.conf
else
    echo "${HELP}"
    exit 1
fi

# ---------------------------
BATCH=$1

ONNX=fast_${QTYPE}.onnx

if [ -f "build/data/bert_tf_v1_1_large_fp32_384_v2/$ONNX" ]; then
    echo "build/data/bert_tf_v1_1_large_fp32_384_v2/$ONNX exists."
else
    echo "Quantize model.onnx to data type ${QTYPE} ..."
    python3 quantize_onnx.py --precision ${QTYPE}
fi

if [ -f "./audit.config" ]; then
    rm ./audit.config
fi

for SCENARIO in Offline SingleStream
do
    if [ "${SCENARIO}" = "SingleStream" ]; then
        BATCH=1
    fi

    echo "SCENARIO=${SCENARIO} BatchSize=${BATCH} NumThreads=${NUM_THREADS}"

    RESULT_DIR=results/fast_${QTYPE}_batch_${BATCH}_${NUM_THREADS}/${SCENARIO}
    AUDIT_DIR=build/compliance_output/fast_${QTYPE}_batch_${BATCH}_${NUM_THREADS}/${SCENARIO}

    OUT_DIR=${RESULT_DIR}/accuracy
    if [ -d "${OUT_DIR}" ]; then
        echo "Skip ${SCENARIO} tests since directory exists ${OUT_DIR}"
    else
        mkdir -p ${OUT_DIR}
        echo "Start accuracy test for scenario=${SCENARIO} batch_size=${BATCH}..."
        python3 run.py --scenario ${SCENARIO} --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime --accuracy
        python3 accuracy-squad.py > ${OUT_DIR}/accuracy.txt
        mv build/logs/* ${OUT_DIR}/
    fi

    OUT_DIR=${RESULT_DIR}/performance/run_1
    if [ -d "${OUT_DIR}" ]; then
        echo "Skip ${SCENARIO} tests since directory exists ${OUT_DIR}"
    else
        mkdir -p ${OUT_DIR}
        echo "Start performance test for scenario=${SCENARIO} batch_size=${BATCH}..."
        python3 run.py --scenario ${SCENARIO} --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime
        mv build/logs/* ${OUT_DIR}/
    fi

    TEST_NAME="TEST01"
    cp ../../compliance/nvidia/${TEST_NAME}/bert/audit.config ./audit.config
    echo "Start ${TEST_NAME} for scenario=${SCENARIO} batch_size=${BATCH}..."
    python3 run.py --backend=onnxruntime --scenario=${SCENARIO} --batch_size ${BATCH} --onnx_filename $ONNX
    python3 ../../compliance/nvidia/${TEST_NAME}/run_verification.py --results_dir ${RESULT_DIR}/ --compliance_dir build/logs/ --output_dir ${AUDIT_DIR}/

    TEST_NAME="TEST05"
    cp ../../compliance/nvidia/${TEST_NAME}/audit.config ./audit.config
    echo "Start ${TEST_NAME} for scenario=${SCENARIO} batch_size=${BATCH}..."
    python3 run.py --backend=onnxruntime --scenario=${SCENARIO} --batch_size ${BATCH} --onnx_filename $ONNX
    python3 ../../compliance/nvidia/${TEST_NAME}/run_verification.py --results_dir ${RESULT_DIR}/ --compliance_dir build/logs/ --output_dir ${AUDIT_DIR}/

    # Must remove the audit.config before next test
    rm ./audit.config

    SAVE_DIR=results_b$1_t${NUM_THREADS}_r$2
    mkdir -p ${SAVE_DIR}/results/${TEST_BOX}/bert-99/${SCENARIO}/
    mv ${RESULT_DIR}/* ${SAVE_DIR}/results/${TEST_BOX}/bert-99/${SCENARIO}/

    mkdir -p ${SAVE_DIR}/compliance/${TEST_BOX}/bert-99/${SCENARIO}/
    mv ${AUDIT_DIR}/* ${SAVE_DIR}/compliance/${TEST_BOX}/bert-99/${SCENARIO}/

    if [ "${SCENARIO}" = "SingleStream" ]; then
        grep "90th percentile latency (ns)" ${SAVE_DIR}/results/${TEST_BOX}/bert-99/${SCENARIO}/performance/run_1/mlperf_log_summary.txt
    else
        grep "Samples per second:" ${SAVE_DIR}/results/${TEST_BOX}/bert-99/${SCENARIO}/performance/run_1/mlperf_log_summary.txt
    fi

    grep "f1" ${SAVE_DIR}/results/${TEST_BOX}/bert-99/${SCENARIO}/accuracy/accuracy.txt
    grep 'TEST' ${SAVE_DIR}/compliance/${TEST_BOX}/bert-99/${SCENARIO}/TEST01/verify_accuracy.txt | sed 's/^/TEST01 accuracy /'
    grep 'TEST' ${SAVE_DIR}/compliance/${TEST_BOX}/bert-99/${SCENARIO}/TEST01/verify_performance.txt | sed 's/^/TEST01 performance /'
    grep 'TEST' ${SAVE_DIR}/compliance/${TEST_BOX}/bert-99/${SCENARIO}/TEST05/verify_performance.txt | sed 's/^/TEST05 performance /'
done

echo "Done. Results of batch size $1 of run $2 is saved to ${SAVE_DIR}"
