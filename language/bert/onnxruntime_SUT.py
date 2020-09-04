# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import onnxruntime
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL

class BERT_ONNXRuntime_SUT():
    def __init__(self, args):
        self.profile = args.profile
        self.options = onnxruntime.SessionOptions()
        self.options.enable_profiling = args.profile

        model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/" + args.onnx_filename
        self.input_mask_name = "attention_mask" if (args.onnx_filename == "bert_large_v1_1_fake_quant.onnx") else "input_mask"
        self.segment_ids_name = "token_type_ids" if (args.onnx_filename == "bert_large_v1_1_fake_quant.onnx") else "segment_ids"
        self.sess = onnxruntime.InferenceSession(model_path, self.options)

        print(f"Loading ONNX model {model_path}...")

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL()

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)

            actual_seq_length = sum(eval_features.input_mask)
            fd = {
                "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :actual_seq_length],
                self.input_mask_name: np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :actual_seq_length],
                self.segment_ids_name: np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :actual_seq_length]
            }
           
            scores = self.sess.run([o.name for o in self.sess.get_outputs()], fd)
            output = np.stack(scores, axis=-1)[0]

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        if self.profile:
            print("ONNX runtime profile dumped to: '{}'".format(self.sess.end_profiling()))
        print("Finished destroying SUT.")

def get_onnxruntime_sut(args):
    return BERT_ONNXRuntime_SUT(args)
