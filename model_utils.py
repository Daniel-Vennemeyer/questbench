# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for calling models."""

from concurrent import futures
import json
import os
from typing import Dict, List

import torch
import transformers


ThreadPoolExecutor = futures.ThreadPoolExecutor
pipeline = transformers.pipeline


# Initialize local Llama 70B pipeline
llama_pipeline = pipeline(
    "text-generation",
    model="llama-70b",
    device_map="auto",
    torch_dtype=torch.float16,
)


def load_cache_file(cache_file):
  cache = {}
  if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
      for line in f:
        line = json.loads(line)
        cache[line["prompt"]] = line["completion"]
  return cache


def jsonify_prompt(prompt):
  return json.dumps(prompt)


def model_call_wrapper(
    model_name,
    model_url,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, str],
    parallel_model_calls: bool,
) -> List[str]:
    """Wrapper for calling the local Llama 70B model."""
    if not batch_messages:
        return []
    responses = []
    for messages in batch_messages:
        # Concatenate message contents as prompt
        prompt = "".join(message["content"] for message in messages)
        output = llama_pipeline(
            prompt,
            max_new_tokens=generation_config.get("max_tokens", 512),
            temperature=generation_config.get("temperature", 0.0),
        )
        # Extract generated text
        responses.append(output[0]["generated_text"])
    return responses


def cached_generate(
    batch_prompts,
    model_name,
    model_url,
    cache,
    cache_file,
    generation_config,
    parallel_model_calls,
):
  """Generate a batch of responses from a model, caching responses.

  Args:
    batch_prompts: The batch of prompts.
    model_name: The name of the model to generate from.
    model_url: The URL of the model to generate from.
    cache: cache of LLM responses.
    cache_file: cache file of LLM responses.
    generation_config: generation config for LLM
    parallel_model_calls: whether to make parallel calls to the model

  Returns:
    The batch of responses and the cost of the generation.
  """
  if model_name.startswith("o1"):
    for prompt in batch_prompts:
      for t, turn in enumerate(prompt):
        if turn["role"] == "system":
          prompt[t]["role"] = "user"
    generation_config = {}
  if cache is None:
    return model_call_wrapper(
        batch_prompts,
        model_name,
        model_url,
        generation_config=generation_config,
        parallel_model_calls=parallel_model_calls,
    )
  new_batch_prompts = []
  for prompt in batch_prompts:
    # jsonify prompt
    jsonified_prompt = jsonify_prompt(prompt)
    if jsonified_prompt not in cache:
      new_batch_prompts.append(prompt)
  batch_responses = model_call_wrapper(
      model_name,
      model_url,
      batch_messages=new_batch_prompts,
      generation_config=generation_config,
      parallel_model_calls=parallel_model_calls,
  )
  for prompt, response in zip(new_batch_prompts, batch_responses):
    jsonified_prompt = jsonify_prompt(prompt)
    cache[jsonified_prompt] = response
    with open(cache_file, "a") as f:
      f.write(
          json.dumps({
              "prompt": jsonified_prompt,
              "completion": cache[jsonified_prompt],
          })
          + "\n"
      )
  # throws error to retry after having saved
  assert len(batch_responses) == len(new_batch_prompts)
  batch_responses = []
  cost = 0.0
  for prompt in batch_prompts:
    jsonified_prompt = jsonify_prompt(prompt)
    text_output = cache[jsonified_prompt]
    batch_responses.append(text_output)
  return batch_responses, cost
