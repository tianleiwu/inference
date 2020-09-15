import os
import argparse
import onnx
from onnxruntime_tools.transformers.onnx_model_bert import BertOnnxModel, BertOptimizationOptions
from onnxruntime_tools.transformers.optimizer import optimize_model
from onnxruntime_tools.transformers.quantize_helper import QuantizeHelper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_onnx",
                        type=str,
                        default='build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx',
                        help="original onnx model filename")
    parser.add_argument("--prefix", type=str, default='fast', help="onnx model filename prefix")
    parser.add_argument("--input_int64", action="store_true", help="use int64 as data type of input")
    parser.add_argument("--no_fast_gelu", action="store_true", help="do not use gelu approximation")
    parser.add_argument("--no_raw_mask", action="store_true", help="do not use raw attention mask")
    parser.add_argument("--precision",
                        type=str,
                        default='fp16',
                        choices=['fp32', 'fp16', 'int8'],
                        help="precision of model")
    args = parser.parse_args()
    return args


def set_dynamic_axes(model, dynamic_batch_dim='batch_size', dynamic_seq_len='max_seq_len'):
    """
    Update input and output shape to use dynamic axes.
    """
    for input in model.graph.input:
        dim_proto = input.type.tensor_type.shape.dim[0]
        dim_proto.dim_param = dynamic_batch_dim
        if dynamic_seq_len is not None:
            dim_proto = input.type.tensor_type.shape.dim[1]
            dim_proto.dim_param = dynamic_seq_len

    for output in model.graph.output:
        dim_proto = output.type.tensor_type.shape.dim[0]
        dim_proto.dim_param = dynamic_batch_dim
        if dynamic_seq_len is not None:
            dim_proto = output.type.tensor_type.shape.dim[1]
            dim_proto.dim_param = dynamic_seq_len


'''
Create model with dynamic axes for sequence length, and with an option to merge two outputs into one.
'''


def create_dynamic_model(input_onnx_path, output_onnx_path):
    model = onnx.load_model(input_onnx_path, format=None, load_external_data=True)
    bert_model = BertOnnxModel(model, 16, 1024)

    set_dynamic_axes(bert_model.model, dynamic_batch_dim='batch_size', dynamic_seq_len='max_seq_len')

    bert_model.save_model_to_file(output_onnx_path)


def create_optimized_model(input_onnx_path,
                           output_onnx_path,
                           fast_gelu=True,
                           raw_mask=True,
                           input_int32=True,
                           fp16=True):
    optimization_options = BertOptimizationOptions('bert')
    optimization_options.enable_gelu_approximation = fast_gelu
    if not raw_mask:
        optimization_options.use_raw_attention_mask(False)

    optimizer = optimize_model(input_onnx_path,
                               "bert",
                               num_heads=16,
                               hidden_size=1024,
                               opt_level=1,
                               optimization_options=optimization_options,
                               use_gpu=False,
                               only_onnxruntime=False)

    if input_int32:
        optimizer.change_input_to_int32()

    if fp16:
        optimizer.convert_model_float32_to_float16()

    optimizer.save_model_to_file(output_onnx_path)


def main():
    args = get_args()

    onnx_dir = os.path.dirname(args.input_onnx)
    temp_onnx = os.path.join(onnx_dir, f"{args.prefix}.onnx")
    output_fp32_onnx = os.path.join(onnx_dir, f"{args.prefix}_fp32.onnx")
    output_fp16_onnx = os.path.join(onnx_dir, f"{args.prefix}_fp16.onnx")
    output_int8_onnx = os.path.join(onnx_dir, f"{args.prefix}_int8.onnx")

    create_dynamic_model(args.input_onnx, temp_onnx)

    if args.precision == "fp16":
        if not os.path.exists(output_fp16_onnx):
            create_optimized_model(temp_onnx,
                                   output_fp16_onnx,
                                   fast_gelu=not args.no_fast_gelu,
                                   raw_mask=not args.no_raw_mask,
                                   input_int32=not args.input_int64,
                                   fp16=True)
            print(f"Created optimized fp16 model: {output_fp16_onnx}")
        else:
            print(f"Skip creating existed model: {output_fp16_onnx}")
    else:
        if not os.path.exists(output_fp32_onnx):
            create_optimized_model(temp_onnx,
                                   output_fp32_onnx,
                                   fast_gelu=not args.no_fast_gelu,
                                   raw_mask=not args.no_raw_mask,
                                   input_int32=not args.input_int64,
                                   fp16=False)
            print(f"Created optimized fp32 model: {output_fp32_onnx}")
        else:
            print(f"Skip creating existed model: {output_fp32_onnx}")

    if args.precision == 'int8':
        QuantizeHelper.quantize_onnx_model(output_fp32_onnx, output_int8_onnx, use_external_data_format=False)
        print(f"Created optimized int8 model: {output_int8_onnx}")

    os.remove(temp_onnx)


if __name__ == "__main__":
    main()
