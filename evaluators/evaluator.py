# Copyright 2025 DeepMind Technologies Limited
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

"""Base class for evaluators."""

from model_utils import GPT_COSTS
from model_utils import load_cache_file


class Evaluator:
  """Base class for evaluators.

  Attributes:
    model_name: name of LLM to evaluate
    generation_config: generation config for LLM
    model_url: model url of LLM
    cache: cache of LLM responses
    cache_file: cache file of LLM responses
    use_cot: whether to use CoT or not
    fs_samples: number of few-shot samples to use
    eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
  """

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mc",
  ):
    self.model_name = model_name
    self.generation_config = {
        "temperature": 0.0,
        "max_completion_tokens": 512,
    }
    if "gemini" in self.model_name:
      self.model_url = self.model_name
    elif self.model_name == "gemma_2b":
      self.model_url = "google/gemma-2-2b-it"
    elif self.model_name == "gemma_27b":
      self.model_url = "google/gemma-2-27b-it"
    elif self.model_name == "gemma_9b":
      self.model_url = "google/gemma-2-9b-it"
    elif self.model_name in GPT_COSTS:
      self.generation_config = {
          "temperature": 0.0,
          "max_completion_tokens": 512,
          "top_p": 1.0,
          "frequency_penalty": 0.0,
          "presence_penalty": 0.0,
      }
      self.model_url = "https://api.openai.com/v1/chat/completions"
    self.cache = cache
    self.cache_file = cache_file
    if cache is None and cache_file is not None:
      self.cache = load_cache_file(cache_file)
    self.use_cot = use_cot
    self.fs_samples = fs_samples
    self.eval_mode = eval_mode
