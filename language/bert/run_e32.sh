# Run MLPerf on Azure VM (prefer E*s_v4 with VNNI)
# Usage: sh run_e32.sh [batch_size] [run_id]
export OMP_NUM_THREADS=14
export OMP_WAIT_POLICY=ACTIVE

QTYPE=int8
BATCH=$1

cp user_int8.conf user.conf

echo "BatchSize=${BATCH}"

ONNX=fast_${QTYPE}.onnx

if [ -f "build/data/bert_tf_v1_1_large_fp32_384_v2/$ONNX" ]; then
    echo "build/data/bert_tf_v1_1_large_fp32_384_v2/$ONNX exists."
else
    echo "Quantize to data type ${QTYPE} ..."
    python3 quantize_onnx.py --precision ${QTYPE}
fi

if [ -f "./audit.config" ]; then
    rm ./audit.config
fi

for SCENARIO in "Offline" "SingleStream"
do
    OUT_DIR=results/fast_${QTYPE}_batch_${BATCH}/${SCENARIO}/accuracy
    if [ -d "${OUT_DIR}" ]; then
        echo "Skip ${SCENARIO} tests since directory exists ${OUT_DIR}"
    else
        echo "${SCENARIO} accuracy test ..."
        mkdir -p ${OUT_DIR}
        python3 run.py --scenario ${SCENARIO} --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime --accuracy >${OUT_DIR}/stdout.txt
        mv build/logs/* ${OUT_DIR}/
    fi

    OUT_DIR=results/fast_${QTYPE}_batch_${BATCH}/${SCENARIO}/performance/run_1
    if [ -d "${OUT_DIR}" ]; then
        echo "Skip ${SCENARIO} tests since directory exists ${OUT_DIR}"
    else
        mkdir -p ${OUT_DIR}
        python3 run.py --scenario ${SCENARIO} --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime >${OUT_DIR}/stdout.txt
        mv build/logs/* ${OUT_DIR}/
    fi

    cp ~/inference/compliance/nvidia/TEST01/bert/audit.config ./audit.config

    python3 run.py --backend=onnxruntime --scenario=${SCENARIO} --batch_size ${BATCH} --onnx_filename $ONNX

    python3 ~/inference/compliance/nvidia/TEST01/run_verification.py --results_dir results/fast_${QTYPE}_batch_${BATCH}/${SCENARIO}/ --compliance_dir build/logs/ --output_dir build/compliance_output/fast_${QTYPE}_batch_${BATCH}/${SCENARIO}/
    
done

rm ./audit.config

mv results/fast_${QTYPE}_batch_${BATCH}/ results_b${BATCH}_$2/results/E32/bert/
mv build/compliance_output/fast_${QTYPE}_batch_${BATCH}/ results_b${BATCH}_$2/compliance/E32/bert/