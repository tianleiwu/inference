# Test Quantizied ONNX BERT model (prefer VNNI machine)
# For int8 CPU inference, set OMP environment variable properly
#   OMP_NUM_THREADS=16
#   OMP_WAIT_POLICY=ACTIVE
# Use MLPERF_BATCH and MLPERF_DTYPE to control batch size and data type (int8, fp16)

QTYPE=${MLPERF_DTYPE:-int8}
BATCH=${MLPERF_BATCH:-1}
echo "BatchSize=${BATCH}"

ONNX=fast_${QTYPE}.onnx

if [ -f "build/data/bert_tf_v1_1_large_fp32_384_v2/$ONNX" ]; then
    echo "build/data/bert_tf_v1_1_large_fp32_384_v2/$ONNX exists."
else
    echo "Quantize to data type ${QTYPE} ..."
    python quantize_onnx.py --precision ${QTYPE}
fi

OUT_DIR=results/fast_${QTYPE}_batch_${BATCH}/offline_accuracy
if [ -d "${OUT_DIR}" ]; then
    echo "Skip offline tests since directory exists ${OUT_DIR}"
else
    echo "Offline accuracy test ..."
    mkdir -p ${OUT_DIR}
    python run.py --scenario Offline --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime --accuracy >${OUT_DIR}/stdout.txt
    mv build/logs/* ${OUT_DIR}/
fi

OUT_DIR=results/fast_${QTYPE}_batch_${BATCH}/offline
if [ -d "${OUT_DIR}" ]; then
    echo "Skip offline tests since directory exists ${OUT_DIR}"
else
    mkdir -p ${OUT_DIR}
    OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --scenario Offline --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime >${OUT_DIR}/stdout.txt
    mv build/logs/* ${OUT_DIR}/
fi

OUT_DIR=results/fast_${QTYPE}/singlestream

if [ -d "${OUT_DIR}" ]; then
    echo "Skip single stream tests since directory exists ${OUT_DIR}"
else
    mkdir -p ${OUT_DIR}
    python run.py --scenario SingleStream --onnx_filename $ONNX --backend onnxruntime >${OUT_DIR}/stdout.txt
    mv build/logs/* ${OUT_DIR}/
fi

OUT_DIR=results/fast_${QTYPE}/singlestream_accuracy
if [ -d "${OUT_DIR}" ]; then
    echo "Skip single stream tests since directory exists ${OUT_DIR}"
else
    mkdir -p ${OUT_DIR}
    python run.py --scenario SingleStream --onnx_filename $ONNX --backend onnxruntime --accuracy >${OUT_DIR}/stdout.txt
    mv build/logs/* ${OUT_DIR}/
fi
