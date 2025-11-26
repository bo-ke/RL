# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: ernie_router_environment.py
@time: 2025/11/20 20:09:57
@Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""

import logging
import re
from functools import partial
from typing import Any, Callable, List, Optional, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.rewards import (
    combine_reward_functions,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class ErnieRouterEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    reward_functions: List[dict[str, Any]]  # list of reward functions and their weights


def format_reward(
    ground_truth: str,
    response: str,
    think_tag: str = "think",
) -> tuple[float, Optional[bool]]:
    """Reward the agent when the response follows the format: (.*) <think> (.*) </think> <answer> (.*) </answer>.

    The `think_tag` and `answer_tag` are customizable and must be specified as part of the user COT prompt text file.
    """
    rew = 0.0
    if re.search(rf"[\s\S]*</{think_tag}>", response):
        rew += 0.25  # 0.25 points for having think tags
    if re.search(rf"</{think_tag}>(\n+\d+)", response):
        rew += 0.75  # 0.75 points for having answer tags
    return rew, None


def exact_answer_alphanumeric_reward(
    ground_truth: str,
    response: str,
    think_tag: str = "think",
) -> tuple[float, bool]:
    match = re.search(rf"</{think_tag}>\s+(\d+)", response)
    if match:
        answer = match.group(1)  # '2'
        if ground_truth.lower() in answer.lower():
            return 1.0, True
    return 0.0, False


@ray.remote
class ErnieRouterVerifyWorker:
    def __init__(self, cfg: ErnieRouterEnvConfig) -> None:
        logging.getLogger("ernie_router_verify_worker").setLevel(logging.CRITICAL)
        # this is a simple reward function that rewards the agent for correct answer and correct format
        reward_functions = []
        # loop over all configs
        for reward_func_cfg in cfg["reward_functions"]:
            # get name and weight
            reward_func_name: str = reward_func_cfg["name"]
            reward_func_weight: float = reward_func_cfg["weight"]
            reward_func_kwargs: Optional[dict] = reward_func_cfg.get("kwargs", None)
            reward_func: Callable[[str, str], tuple[float, Optional[bool]]]
            if reward_func_name == "format":
                reward_func = format_reward
            elif reward_func_name == "exact_result":
                reward_func = exact_answer_alphanumeric_reward
            else:
                raise ValueError(f"Invalid reward function: {reward_func_name}")

            # check for additional kwargs
            if reward_func_kwargs is not None:
                reward_func = partial(reward_func, **reward_func_kwargs)

            reward_functions.append((reward_func, reward_func_weight))

        if len(reward_functions) == 0:
            raise ValueError("No reward functions provided")

        # combine the reward functions
        self.verify_func = combine_reward_functions(reward_functions)

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                ret_score, _ = self.verify_func(ground_truth, response)
            except Exception as e:
                ret_score = 0.0
                print(f"Error in verify_func: {e}")
            results.append(float(ret_score))
        return results


class ErnieRouterEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ErnieRouterEnvironment(EnvironmentInterface):
    def __init__(self, cfg: ErnieRouterEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            ErnieRouterVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[ErnieRouterEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the vlm environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the VLM.
            metadata: list[ErnieRouterEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)
        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {"role": "environment", "content": f"Environment: reward: {result}"}
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
