# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""Ernie chat dataset for GRPO and SFT training.

@author: kebo
@contact: kebo01@baidu.com
@version: 1.0
@file: ernie_data.py
@time: 2025/11/18 15:01:14
@Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
"""

import copy
import io
import json
import random
from collections import defaultdict
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
from nemo_rl.data.llm_message_utils import get_formatted_message_log
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
    "ernie_chat_sft_data_processor",
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


def ernie_chat_sft_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_generation_prompt=True,
) -> DatumSpec:
    user_message = datum_dict["messages"]
    user_message.append(datum_dict["candidates"][0][0])

    for dialog in user_message:
        for content in dialog["content"]:
            if content["type"] == "image_url":
                content["type"] = "image"
                content["image"] = resolve_to_image(content.pop("image_url")["url"])

    message_log = get_formatted_message_log(
        user_message,
        tokenizer,
        task_data_spec,
        add_bos_token=False,
        add_eos_token=True,  # TODO
        add_generation_prompt=add_generation_prompt,
    )
    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # 超长样本：创建 fake 数据，带一张空图，loss_multiplier=0
        # 这样 vision encoder 参与计算但不影响训练
        print(
            f"[WARNING] Sample {idx} too long ({length} > {max_seq_length}), replacing with fake image data"
        )
        fake_image = Image.new(
            "RGB", (56, 56), color="white"
        )  # 最小的图像，产生最少的 patch
        fake_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": fake_image},
                    {"type": "text", "text": "hi"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        message_log = get_formatted_message_log(
            fake_message,
            tokenizer,
            task_data_spec,
            add_bos_token=False,
            add_eos_token=True,
            add_generation_prompt=False,
        )
        length = sum(len(m["token_ids"]) for m in message_log)
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


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
    extra_env_info = {
        "ground_truth": datum_dict["candidates"][0][0]["content"][0]["text"]
    }
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

    # 使用 processor 直接处理文本和图像，确保返回 pixel_values 和 image_grid_thw
    # 这与 SFT 中 get_formatted_message_log 的处理方式一致
    message: dict = processor(
        text=[string_formatted_dialog],
        images=images if len(images) > 0 else None,
        return_tensors="pt",
        add_special_tokens=False,
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
        if key in message and message[key] is not None:
            model_inputs[key] = PackedTensor(
                message[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )
    ### append to user message
    message_log.append(model_inputs)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        # 超长样本：创建 fake 数据，带一张空图，loss_multiplier=0
        # 这样 vision encoder 参与计算但不影响训练
        print(
            f"[WARNING] Sample {idx} too long ({length} > {max_seq_length}), replacing with fake image data"
        )
        fake_image = Image.new(
            "RGB", (56, 56), color="white"
        )  # 最小的图像，产生最少的 patch
        fake_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": fake_image},
                    {"type": "text", "text": "hi"},
                ],
            },
        ]
        string_formatted_dialog = processor.apply_chat_template(
            fake_message,
            tokenize=False,
            add_generation_prompt=True,
        )
        message: dict = processor(
            text=[string_formatted_dialog],
            images=[fake_image],
            return_tensors="pt",
            add_special_tokens=False,
        )
        model_inputs = {
            "role": "user",
            "token_ids": message["input_ids"][0],
        }
        multimodal_keys = get_multimodal_keys_from_processor(processor)
        for key in multimodal_keys:
            if key in message and message[key] is not None:
                model_inputs[key] = PackedTensor(
                    message[key], dim_to_pack=get_dim_to_pack_along(processor, key)
                )
        message_log = [model_inputs]
        length = sum(len(m["token_ids"]) for m in message_log)
        loss_multiplier = 0.0
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": [fake_image],
        }
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


class DataSingleSource:
    def __init__(self, src_id, filepath, task_name, data_type, shuffle=True, seed=42):
        self.src_id = src_id
        self.task_name = task_name
        self.data_type = data_type
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
                    "candidates": line["candidates"],
                    "task_name": self.task_name,
                }
            )
        f.close()
        return ds

    def __getitem__(self, idx):
        return copy.deepcopy(self.data[idx])

    def sampler(self):
        idxs = list(range(len(self)))
        if self.shuffle:
            self.rng.shuffle(idxs)
        while True:
            for idx in idxs:
                yield idx
            idxs = list(range(len(self)))
            if self.shuffle:
                self.rng.shuffle(idxs)


class ReeaoChatData:
    def __init__(
        self,
        data_config_file,
        batch_size,
        num_samples=None,
        num_epoch=1,
        shuffle_data=True,
        seed=42,
    ):
        self.rng = random.Random(seed)
        self.seed = seed
        self.data_config = json.load(open(data_config_file, "r"))
        self.shuffle_data = shuffle_data
        self.num_epoch = num_epoch
        self.batch_size = batch_size

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
                f"loading data at {src_id} from {data_config['filepath']}, normalized_weight: {weight}, type: {data_config.get('data_type', 'lm')}"
            )
            self.data_source[src_id] = {
                "weight": weight,
                "data": DataSingleSource(
                    src_id,
                    data_config["filepath"],
                    data_config.get("task_name", "ernie_router_chat"),
                    data_config.get("data_type", "lm"),
                    self.shuffle_data,
                    self.seed + src_id,
                ),
            }

    def _build_data_seq(self):
        """Build batch-aligned data sequence.

        Strategy:
        1. Calculate sample count for each source based on weight.
        2. Group samples by data_type.
        3. Split into batches within each data_type.
        4. Shuffle batches to mix different data types.
        5. Flatten to final data_seq.
        """
        print("Start building batch-aligned data sequence...")

        # 1. 计算每个源需要多少个样本
        src_counts = {}
        # 先分配整数部分
        allocated_samples = 0
        for src_id in self.data_source:
            count = int(self.num_samples * self.data_source[src_id]["weight"])
            src_counts[src_id] = count
            allocated_samples += count

        # 补齐舍入误差导致的剩余样本，简单加给第一个源
        if allocated_samples < self.num_samples:
            src_counts[list(self.data_source.keys())[0]] += (
                self.num_samples - allocated_samples
            )

        # 2. 按 data_type 分组收集样本索引
        # 结构: {'lm': [(src_id, idx), ...], 'vqa': [(src_id, idx), ...]}
        type_samples_pool = defaultdict(list)

        for src_id, count in src_counts.items():
            dataset_obj = self.data_source[src_id]["data"]
            dtype = dataset_obj.data_type

            # 从生成器中取出指定数量的样本索引
            for _ in range(count):
                type_samples_pool[dtype].append((src_id, next(dataset_obj.index_gen)))

        # 3. 组装 Batch
        all_batches = []

        for dtype, samples in type_samples_pool.items():
            # 在同一类型内部 shuffle 样本顺序
            if self.shuffle_data:
                self.rng.shuffle(samples)

            # 切分 Batch
            # 注意：如果剩余样本不足一个 batch，通常为了保证类型纯度，可以丢弃或者保留。
            # 这里选择保留并作为一个不完整的 batch，因为后续 shuffle batch 后，它是一个独立的 batch，不会混入其他类型。
            # 这里的关键是 DataLoader 必须是 sequential 的，不能再次 shuffle 样本索引。
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)

        # 4. Shuffle Batch 顺序
        if self.shuffle_data:
            print(
                f"Shuffling {len(all_batches)} batches to mix different data types..."
            )
            self.rng.shuffle(all_batches)

        # 5. 展平
        self.data_seq = []
        for batch in all_batches:
            self.data_seq.extend(batch)
        # 修正 num_samples (可能因为丢弃/补齐略有变化，或者保持一致)
        self.num_samples = len(self.data_seq)
        assert self.num_samples % self.batch_size == 0, (
            self.num_samples,
            len(all_batches),
        )
        print(
            f"Data sequence built. Total samples: {self.num_samples}, Total batches: {len(all_batches)}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= len(self.data_seq):
            # 防止越界
            idx = idx % len(self.data_seq)

        src_id, dat_index_id = self.data_seq[idx]
        # print(f"get_item-{idx}, type: {self.data_source[src_id]['data'].data_type}")
        return self.data_source[src_id]["data"][dat_index_id]


class ErnieChatDataset:
    def __init__(
        self,
        train_data_config,
        batch_size,
        train_num_samples=None,
        val_data_path: Optional[str] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        num_epoch=1,
        shuffle=True,
        seed=42,
    ):
        self.task_name = "ernie_router_chat"
        # store the formatted dataset
        # 注意：外部使用 DataLoader 时，shuffle 必须设为 False，否则这里的 Batch 排列会被打乱
        train_ds = ReeaoChatData(
            train_data_config,
            batch_size=batch_size,
            num_samples=train_num_samples,
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
                    "candidates": line["candidates"],
                    "task_name": self.task_name,
                }
            )
        f.close()
        return ds
