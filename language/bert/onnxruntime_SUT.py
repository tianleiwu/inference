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

import psutil
import mlperf_loadgen as lg
import numpy as np
import onnxruntime
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL


class BERT_ONNXRuntime_SUT():
    def __init__(self, args):
        self.profile = args.profile
        options = onnxruntime.SessionOptions()
        options.enable_profiling = args.profile
        #options.inter_op_num_threads = 2

        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            options.intra_op_num_threads = psutil.cpu_count(logical=False)
        else:
            providers = ['CPUExecutionProvider']

        model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/" + args.onnx_filename
        print(f"Loading ONNX model {model_path}...")
        self.sess = onnxruntime.InferenceSession(model_path, options, providers=providers)

        self.input_dtype = np.int32 if self.sess.get_inputs()[0].type == 'tensor(int32)' else np.int64

        self.input_mask_name = self.sess.get_inputs()[1].name
        assert self.input_mask_name in ["attention_mask", "input_mask"]

        self.segment_ids_name = self.sess.get_inputs()[2].name
        assert self.segment_ids_name in ["token_type_ids", "segment_ids"]

        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.batch_size = args.batch_size
        assert self.batch_size >= 1

        self.processed_samples = 0

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL()

    def report(self, sample_id, output):
        response_array = array.array("B", output.tobytes())
        bi = response_array.buffer_info()
        response = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
        lg.QuerySamplesComplete([response])

    def process_sample(self, eval_features, actual_seq_length, sample_id):
        input = {
            "input_ids":
            np.array(eval_features.input_ids).astype(self.input_dtype)[np.newaxis, :actual_seq_length],
            self.input_mask_name:
            np.array(eval_features.input_mask).astype(self.input_dtype)[np.newaxis, :actual_seq_length],
            self.segment_ids_name:
            np.array(eval_features.segment_ids).astype(self.input_dtype)[np.newaxis, :actual_seq_length]
        }

        scores = self.sess.run(self.output_names, input)
        output = np.stack(scores, axis=-1)[0]

        self.report(sample_id, output)

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        if num_samples == 1 or self.batch_size == 1:
            for i in range(num_samples):
                eval_features = self.qsl.get_features(query_samples[i].index)
                actual_seq_length = sum(eval_features.input_mask)
                self.process_sample(eval_features, actual_seq_length, query_samples[i].id)
        else:
            actual_lengths = []
            for i in range(num_samples):
                eval_features = self.qsl.get_features(query_samples[i].index)
                actual_seq_length = sum(eval_features.input_mask)
                actual_lengths.append(actual_seq_length)
            lengths = np.array(actual_lengths)
            sort_index = np.argsort(lengths)

            total_batches = int((num_samples + self.batch_size - 1) / self.batch_size)
            for j in range(total_batches):
                indices = sort_index[self.batch_size * j:self.batch_size * (j + 1)]
                samples = [self.qsl.get_features(query_samples[i].index) for i in indices]
                sample_ids = [query_samples[i].id for i in indices]
                take_lengths = np.take(lengths, indices)
                max_lengths = np.max(take_lengths)

                input_ids = np.zeros((len(samples), max_lengths), dtype=self.input_dtype)
                input_mask = np.zeros((len(samples), max_lengths), dtype=self.input_dtype)
                segment_ids = np.zeros((len(samples), max_lengths), dtype=self.input_dtype)
                for k, sample in enumerate(samples):
                    length = take_lengths[k]
                    input_ids[k, :length] = sample.input_ids[:length]
                    input_mask[k, :length] = sample.input_mask[:length]
                    segment_ids[k, :length] = sample.segment_ids[:length]

                input = {"input_ids": input_ids, self.input_mask_name: input_mask, self.segment_ids_name: segment_ids}

                scores = self.sess.run(self.output_names, input)
                stack_scores = np.stack(scores, axis=-1)

                for k in range(len(samples)):
                    output = stack_scores[k]
                    self.report(sample_ids[k], output)

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
