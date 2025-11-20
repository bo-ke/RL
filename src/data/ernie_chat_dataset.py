# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: ernie_data.py
@time: 2025/11/18 15:01:14
@Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
import logging
from typing import Any, Optional
import argparse
import base64
import os
import json
import pprint
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import requests
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
)
from nemo_rl.data import DataConfig

import io
from PIL import Image
from paimon import PaimonBosClient
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
        logging.info(f"{url} Not Found")
        return None


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
        for content in dialog['content']:
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
        "role": "user", # vllm envrioment reward step的时候有用
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



class ErnieChatDataset:
    
    def __init__(self,
                 filelist,
                 ratio_file=None,
                 val_data_path: Optional[str] = None,
                 train_split: Optional[str] = None,
                 val_split: Optional[str] = None):
        train_ds_dict = {}
        self.task_name = "ernie_router_chat"
        for partid, train_data_path in enumerate([i.strip() for i in open(filelist, "r").readlines()]):
            train_ds_dict[partid] = self.load_dataset_from_path(train_data_path, train_split)
        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds_dict[0],
            "validation": None,
        }
        self.task_spec = TaskDataSpec(task_name=self.task_name)

    def load_dataset_from_path(self, path, split: Optional[str] = None):
        f = open(path, 'r', encoding='utf-8')
        ds = []
        for idx, line in enumerate(f):
            line = json.loads(line)
            ds.append({
                "messages": line["prompt"],
                "tgt": line["candidates"][0][0]['content'][0]['text'],
                "task_name": self.task_name
            })
            if idx > 200:
                break
        return ds
