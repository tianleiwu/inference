for j in 1 2 3
do
   for i in 1 2
   do
      mkdir build/logs_offline_accuracy_fast_${i}_b_${j}
      OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --backend onnxruntime --scenario Offline --onnx_filename fast_${i}_int8.onnx --batch_size ${j} --accuracy >build/logs_offline_accuracy_fast_${i}_b_${j}/stdout.txt 
      mv build/logs/* build/logs_offline_accuracy_fast_${i}_b_${j}/
   done
done

for j in 1 2 3
do
   for i in 1 2
   do
      mkdir build/logs_offline_fast_${i}_b_${j}
      OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --scenario Offline --onnx_filename fast_${i}_int8.onnx --batch_size ${j} --backend onnxruntime > build/logs_offline_fast_${i}_b_${j}/stdout.txt
      mv build/logs/* build/logs_offline_fast_${i}_b_${j}/
   done
done

for i in 1 2
do
   mkdir mkdir build/logs_stream_${i}
   OMP_NUM_THREADS=16 OMP_WAIT_POLICY=ACTIVE python run.py --backend onnxruntime --scenario SingleStream --onnx_filename fast_${i}_int8.onnx > build/logs_stream_${i}/stdout.txt
   mv build/logs/* build/logs_stream_${i}/
done
