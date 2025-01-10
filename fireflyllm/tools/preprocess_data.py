# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/9 15:28
# @File: preprocess_data
# @Email: mlshenkai@163.com
import argparse
import glob
import gzip
import json
import math
import multiprocessing
import time
import sys
import os
# Reference https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py

from typing import Union, List
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from fireflyllm.datasets import indexed_dataset  # copy by megatron-lm


class IdentitySpliter(object):

    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args
        tokenizer_path = args.tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.splitter = IdentitySpliter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_length = 30000
            tokens_list = [
                self.splitter.tokenize(text[i : i + max_length])
                for i in range(0, len(text), max_length)
            ]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line: str):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]

            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = self.tokenizer(sentence, max_length=32000, truncation=True).input_ids
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))

            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(self.tokenizer.eos_token_id)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


# 分割
class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, "r", encoding="utf-8")
        fo = open(output_file_name, "w")

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers)
        split_docs = pool.imap(
            encoder.split, fin, 32
        )

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fo.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fo.close()

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)

        fin = open(input_file_name, "r", encoding="utf-8")

        startup_start = time.time()
        encoder = Encoder(self.args)

        pool = multiprocessing.Pool(
            self.workers
        )
        encoded_docs = pool.imap(
            encoder.encode, fin, 32
        )

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(
                output_prefix, key, level
            )
            output_idx_files[key] = "{}_{}_{}.idx".format(
                output_prefix, key, level
            )

            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key], dtype=indexed_dataset.DType.optimal_dtype(encoder.tokenizer.vocab_size)
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        # for key in builders.keys():
        builders[key].finalize(output_idx_files[key])



def get_args():
    parser = argparse.ArgumentParser()
    # parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument("--tokenizer_path", type=str, required=True)
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                       help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    # if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
    #     print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    if file_name.endswith("/*"):
        file_name = file_name.replace("/*", "/" + args.output_prefix)
    input_file_name = file_name + "_" + str(file_id) + extension  # starcode_1.jsonl
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension  # starcode_ss_1.jsonl
    output_prefix = args.output_prefix + "_" + str(file_id)

    file_names = {
        "partition": input_file_name,
        "sentence_split": sentence_split_file,
        "output_prefix": output_prefix
    }
    return file_names


def check_files_exits(file_names: List[dict], key: str, num_partition: int):
    # 是不是全部都有
    for idx in range(num_partition):
        if not os.path.exists(file_names[idx][key]):
            return False
    return True





def main():
    print("start")
    args = get_args()
    in_ss_out_names = []
    if args.partitions == 1:  # 只有一个文件
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            "partition": args.input,
            "sentence_split": sentence_split_file,
            "output_prefix": args.output_prefix
        }
        in_ss_out_names.append(
            file_names
        )
    else:
        in_file_names = glob.glob(args.input)
        if args.keep_sequential_samples:  # 保持顺序
            total_sample_count = 0
            for file_name in in_file_names:
                with open(file_name, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                    total_sample_count += (fc+1)

            partition_size = math.ceil(total_sample_count / args.partitions)  # 每一块最多多少行 (json)
            print(f"Total total_sample_count ")
        # 创建分块文件名称(路径)信息
        # 有多少个分块，创建多少个
        for idx in range(args.partitions):
            file_names = get_file_name(args, idx)
            in_ss_out_names.append(file_names)
        print(in_ss_out_names)
        partition_present = check_files_exits(in_ss_out_names, key="partition", num_partition=args.partitions)

        split_sentence_present = check_files_exits(in_ss_out_names, key="sentence_split", num_partition=args.partitions)

        if not partition_present and not split_sentence_present:
            # 执行分割，按照每一个文件 partition_size个 json 组成jsonl文档
            partition_input_files = []  # 记录fin
            for idx in range(args.partitions):
                partition_input_file = open(in_ss_out_names[idx]["partition"], "w")
                partition_input_files.append(partition_input_file)

            idx = 0
            if args.keep_sequential_samples:
                line_count = 0
            for in_file_name in in_file_names:
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partition_input_files[idx].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            idx += 1
                    else:
                        idx = (idx + 1) % args.partitions
                fin.close()
            for idx in range(args.partitions):
                partition_input_files[idx].close()
    assert args.workers % args.partitions == 0

    partition = Partition(args, workers=args.workers//args.partitions)

    split_sentence_present = check_files_exits(in_ss_out_names, key="sentence_split", num_partition=args.partitions)


    if args.split_sentences and not split_sentence_present:
        # 对分块后的文档列表进行分块，确保每一行 (json)content的长度不超过设定值，如果超过，则直接分割
        # 多进程处理
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(
                target=partition.split_sentences,
                args=((name["partition"], name["sentence_split"]),)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # if args.partitions == 1:
        #     return
    print("start encoder...")
    # encoder
    processes = []
    input_key = "sentence_split" if args.split_sentences else "partition"
    print("input_key", input_key)
    for name in in_ss_out_names:
        p = multiprocessing.Process(
            target=partition.process_json_file,
            args=((name[input_key], name["output_prefix"]),)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if args.partitions == 1:
        return

        # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    encoder = Encoder(args)
    tokenizer = encoder.tokenizer
    print("combine index and bin")
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            partition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(partition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])

if __name__ == "__main__":
    main()










