# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: ernie_data.py
@time: 2025/11/18 15:01:14
@Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""

import copy
import io
import json
from typing import Any, Optional

from paimon import PaimonBosClient
from PIL import Image
from transformers import AutoProcessor

from nemo_rl.data import DataConfig
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataSpec,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
)

bos = PaimonBosClient("~/paimon_bos_client.yaml")

__all__ = (
    "ErnieChatDataConfig",
    "ErnieChatDataset",
    "ernie_chat_data_processor",
)


def resolve_to_image(url):
    try:
        image = Image.open(io.BytesIO(bos.get_bytes(url)))
        return image
    except Exception as e:
        print(f"{url} Not Found")
        return Image.new("RGB", (224, 224), color="white")


class ErnieChatDataConfig(DataConfig):
    filelist: str


def ernie_chat_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from response_datasets/<dataset_name>.py) into a DatumSpec for the VLM Environment."""
    # depending on the task, format the data differently
    user_message = datum_dict["messages"]
    extra_env_info = {"ground_truth": datum_dict["tgt"]}

    message_log: LLMMessageLogType = []
    # this is the string-tokenized conversation template for the generation policy (for vllm)
    if processor.chat_template is None:
        processor.chat_template = processor.tokenizer.chat_template

    images = []
    for dialog in user_message:
        # for image, video, just append it
        # for text, format the prompt to the problem
        for content in dialog["content"]:
            if content["type"] == "image_url":
                content["type"] = "image"
                content["image"] = resolve_to_image(content.pop("image_url")["url"])
                images.append(content["image"])

    string_formatted_dialog = processor.apply_chat_template(
        user_message,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Qwen model 可以，Ernir模型 当前不支持此接口
    message: dict = processor.apply_chat_template(
        user_message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    # add
    model_inputs = {
        "role": "user",  # vllm envrioment reward step的时候有用
        "token_ids": message["input_ids"][0],
        # "content": user_message, # 没用
        # "string_formatted_dialog": string_formatted_dialog, # 没用
    }
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message:
            model_inputs[key] = PackedTensor(
                message[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )
    ### append to user message
    message_log.append(model_inputs)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        # Treat truncated messages as text only
        vllm_kwargs = {
            "vllm_content": None,
            "vllm_images": [],
        }

        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
            for key, value in chat_message.items():
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
    else:
        # get the prompt content! (use this for vllm-backend that needs formatted dialog and list of images) for the entire conversation
        # add images for vllm serving
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": images,
        }
        # if 'image_pad' in string_formatted_dialog:
        #     if len(images) == 0:
        #         import torch
        #         torch.save({"datum_dict": datum_dict}, f"shit_data_{idx}.pt")
        #         raise ValueError("No images found in dataset.")

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name,
        **vllm_kwargs,
    }
    return output


import random


class DataSingleSource:
    def __init__(self, src_id, filepath, task_name, shuffle=True, seed=42):
        self.src_id = src_id
        self.task_name = task_name  # 添加 task_name 参数
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.data = self._load_dataset_from_path(filepath)
        self.index_gen = self.sampler()

    def __len__(self):
        return len(self.data)

    def _load_dataset_from_path(self, path):
        f = open(path, "r", encoding="utf-8")
        ds = []
        for idx, line in enumerate(f):
            line = json.loads(line)
            ds.append(
                {
                    "messages": line["prompt"],
                    "tgt": line["candidates"][0][0]["content"][0]["text"],
                    "task_name": self.task_name,
                }
            )
        f.close()  # 关闭文件
        return ds

    def __getitem__(self, idx):
        # deepcopy, 后续messages被篡改,第二次采到报错。
        return copy.deepcopy(self.data[idx])

    def sampler(self):
        idxs = list(range(len(self)))  # 转换为 list
        if self.shuffle:
            self.rng.shuffle(idxs)
        while True:
            for idx in idxs:
                yield idx
            # 重新生成索引序列
            idxs = list(range(len(self)))  # 转换为 list
            if self.shuffle:
                self.rng.shuffle(idxs)


class ReeaoChatData:
    def __init__(
        self,
        data_config_file,
        num_samples=None,  # 明确参数名
        num_epoch=1,
        shuffle_data=True,
        seed=42,
    ):
        self.rng = random.Random(seed)
        self.seed = seed  # 保存 seed
        self.data_config = json.load(open(data_config_file, "r"))
        self.shuffle_data = shuffle_data
        self.num_epoch = num_epoch
        self.load_dataset()
        if num_samples is None:
            self.num_samples = sum(
                len(self.data_source[src_id]["data"]) for src_id in self.data_source
            )
        else:
            self.num_samples = num_samples
        self._build_data_seq()
        self._epoch = 0

    def load_dataset(self):
        self.data_source = {}
        total_weight = sum([data_config["prob"] for data_config in self.data_config])
        for src_id, data_config in enumerate(self.data_config):
            weight = data_config["prob"] / total_weight
            print(
                f"loading data at {src_id} from {data_config['filepath']}, normalized_weight: {weight}"
            )
            self.data_source[src_id] = {
                "weight": weight,
                "data": DataSingleSource(
                    src_id,
                    data_config["filepath"],
                    data_config.get("task_name", "ernie_router_chat"),  # 传递 task_name
                    self.shuffle_data,
                    self.seed + src_id,  # 每个数据源用不同的 seed
                ),
            }

    def _build_data_seq(self):  # 添加下划线
        data_seq_placeholder = []
        for src_id in self.data_source:
            count = int(
                self.num_samples * self.data_source[src_id]["weight"]
            )  # 使用 num_samples
            data_seq_placeholder.extend([src_id] * count)

        # 处理舍入误差,确保总数正确
        while len(data_seq_placeholder) < self.num_samples:
            data_seq_placeholder.append(list(self.data_source.keys())[0])
        data_seq_placeholder = data_seq_placeholder[: self.num_samples]

        print("shuffle data sequence")
        if self.shuffle_data:
            self.rng.shuffle(data_seq_placeholder)  # shuffle 是原地操作,不返回值

        data_seq = []
        for src_id in data_seq_placeholder:
            data_seq.append((src_id, next(self.data_source[src_id]["data"].index_gen)))
        self.data_seq = data_seq

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        print(f"get_item-{idx}")
        src_id, dat_index_id = self.data_seq[idx]
        return self.data_source[src_id]["data"][dat_index_id]  # 添加 ['data']


class ErnieChatDataset:
    def __init__(
        self,
        train_data_config,
        train_num_samples=None,  # 训练样本数参数
        val_data_path: Optional[str] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        num_epoch=1,
        shuffle=True,
        seed=42,
    ):
        self.task_name = "ernie_router_chat"
        # store the formatted dataset
        train_ds = ReeaoChatData(
            train_data_config,
            train_num_samples,
            num_epoch=num_epoch,
            shuffle_data=shuffle,
            seed=seed,
        )
        if val_data_path:
            val_ds = self.load_dataset_from_path(val_data_path)
        else:
            val_ds = None
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }
        self.task_spec = TaskDataSpec(task_name=self.task_name)

    def load_dataset_from_path(self, path, split: Optional[str] = None):
        f = open(path, "r", encoding="utf-8")
        ds = []
        for idx, line in enumerate(f):
            line = json.loads(line)
            ds.append(
                {
                    "messages": line["prompt"],
                    "tgt": line["candidates"][0][0]["content"][0]["text"],
                    "task_name": self.task_name,
                }
            )
        f.close()
        return ds
