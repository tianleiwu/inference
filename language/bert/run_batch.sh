# Test Quantizied ONNX BERT model (prefer VNNI machine)
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

OUT_DIR=build/fast_${QTYPE}_batch_${BATCH}
if [ -f "$OUT_DIR" ]; then
    echo "Skip offline tests since directory exists ${OUT_DIR}"
else
    mkdir $OUT_DIR

    echo "Offline accuracy test ..."
    mkdir ${OUT_DIR}/offline_accuracy
    OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --scenario Offline --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime --accuracy >${OUT_DIR}/offline_accuracy/stdout.txt
    mv build/logs/* ${OUT_DIR}/offline_accuracy/

    mkdir ${OUT_DIR}/offline
    OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --scenario Offline --onnx_filename $ONNX --batch_size ${BATCH} --backend onnxruntime >${OUT_DIR}/offline/stdout.txt
    mv build/logs/* ${OUT_DIR}/offline/
fi

OUT_DIR=build/fast_${QTYPE}
if [ -f "$OUT_DIR" ]; then
    echo "Skip single stream tests since directory exists ${OUT_DIR}"
else
    mkdir $OUT_DIR

    mkdir ${OUT_DIR}/singlestream
    OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --scenario SingleStream --onnx_filename $ONNX --backend onnxruntime >${OUT_DIR}/singlestream/stdout.txt
    mv build/logs/* ${OUT_DIR}/singlestream/

    mkdir ${OUT_DIR}/singlestream_accuracy
    OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --scenario SingleStream --onnx_filename $ONNX --backend onnxruntime --accuracy >${OUT_DIR}/singlestream_accuracy/stdout.txt
    mv build/logs/* ${OUT_DIR}/singlestream_accuracy/
fi
