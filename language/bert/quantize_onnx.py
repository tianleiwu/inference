import onnx
from onnxruntime_tools.transformers.onnx_model_bert import BertOnnxModel, BertOptimizationOptions
from onnxruntime_tools.transformers.optimizer import optimize_model
from onnxruntime_tools.transformers.quantize_helper import QuantizeHelper


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

def create_dynamic_model(onnx_dir):
    model = onnx.load_model(onnx_dir + "model.onnx",
                            format=None, load_external_data=True)
    bert_model = BertOnnxModel(model, 16, 1024)
    set_dynamic_axes(
        bert_model.model, dynamic_batch_dim='batch_size', dynamic_seq_len='max_seq_len')
    bert_model.save_model_to_file("model_dynamic.onnx")


def create_quantized_model(onnx_dir, use_raw_mask=True):
    optimization_options = BertOptimizationOptions('bert')
    if use_raw_mask:
        optimization_options.use_raw_attention_mask()

    optimizer = optimize_model(onnx_dir + "model_dynamic.onnx",
                               "bert",
                               num_heads=16,
                               hidden_size=1024,
                               opt_level=1,
                               optimization_options=optimization_options,
                               use_gpu=False,
                               only_onnxruntime=False)
    #optimizer.change_input_to_int32()
    optimizer.save_model_to_file(onnx_dir + "model_dynamic_opt.onnx")
    QuantizeHelper.quantize_onnx_model(onnx_dir + "model_dynamic_opt.onnx",
                                       onnx_dir + "model_dynamic_quantized.onnx", use_external_data_format=False)

onnx_dir = "build/data/bert_tf_v1_1_large_fp32_384_v2/"
create_dynamic_model(onnx_dir)
create_quantized_model(onnx_dir, use_raw_mask=True)