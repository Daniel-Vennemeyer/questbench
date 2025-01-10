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

"""Script to evaluate LLMs on QuestBench domains."""

import argparse
import os
from evaluators.gsm import GSMEvaluator
from evaluators.planning import PlanningEvaluator
from evaluators.simple_logic import SimpleLogicEvaluator
import pandas as pd


def main(user_args) -> None:
  domain_main_name = user_args.domain_name.split("_")[0]
  use_cot = False
  fs_samples = 0
  use_phys_constraints = False
  if user_args.prompt_mode == "cot":
    use_cot = True
  elif user_args.prompt_mode == "phys":
    use_phys_constraints = True
  elif user_args.prompt_mode.startswith("fs"):
    fs_samples = int(user_args.prompt_mode[2:])

  # Make directories for results and cache
  if not os.path.exists(user_args.results_dir):
    os.makedirs(user_args.results_dir)
  cache_dir = os.path.join(user_args.results_dir, "cache")
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  data_file_base_name = os.path.splitext(os.path.basename(user_args.data_file))[
      0
  ]
  output_file_name = f"{user_args.model_name}-{user_args.domain_name}-{user_args.eval_mode}-{user_args.prompt_mode}-{data_file_base_name}"
  cache_file = os.path.join(cache_dir, f"{output_file_name}.jsonl")
  output_file = os.path.join(user_args.results_dir, f"{output_file_name}.csv")
  print("Loading Evaluator")
  if domain_main_name == "SL":
    evaluator = SimpleLogicEvaluator(
        user_args.model_name,
        cache_file=cache_file,
        use_cot=use_cot,
        fs_samples=fs_samples,
        eval_mode=user_args.eval_mode,
    )
    prompt_file = os.path.join(
        user_args.data_dir,
        "Logic-Q/simplelogic_heldout_1k_prompts.csv",
    )
  elif domain_main_name == "GSM":
    assert user_args.domain_name.split("_")[1] in ["csp", "verbal"]
    evaluator = GSMEvaluator(
        user_args.model_name,
        cache_file=cache_file,
        use_cot=use_cot,
        fs_samples=fs_samples,
        verbal_questions="verbal" in user_args.domain_name,
        eval_mode=user_args.eval_mode,
    )
    if user_args.domain_name.split("_")[1] == "csp":
      prompt_file = os.path.join(
          user_args.data_dir,
          "GSM-Q/gsm_CSP_heldout_pilot_prompts.csv",
      )
    else:
      prompt_file = os.path.join(
          user_args.data_dir,
          "GSM-Q/gsm_verbal_heldout_pilot_prompts.csv",
      )
  elif domain_main_name == "Planning":
    evaluator = PlanningEvaluator(
        user_args.model_name,
        domain_file=os.path.join(
            user_args.data_dir,
            "Planning-Q/task_pddls/blocks/domain.pddl",
        ),
        task_file_pattern=os.path.join(
            user_args.data_dir,
            "Planning-Q/task_pddls/blocks/task*.pddl",
        ),
        cache_file=cache_file,
        use_cot=use_cot,
        use_phys_constraints=use_phys_constraints,
        fs_samples=fs_samples,
        eval_mode=user_args.eval_mode,
    )
    prompt_file = os.path.join(
        user_args.data_dir,
        "Planning-Q/planning_heldout_prompts.csv",
    )
  else:
    raise SystemExit(f"Unknown domain: {domain_main_name}")

  print("Loading Data")
  data_file = user_args.data_file
  with open(data_file, "r") as f:
    data = pd.read_csv(f)
  prompt_data = None
  if os.path.exists(prompt_file):
    with open(prompt_file, "r") as f:
      prompt_data = pd.read_csv(f)

  print("Starting Evaluation")
  results = evaluator.evaluate_data(data, prompt_data)

  with open(output_file, "w") as wf:
    results.to_csv(wf)
  print(f"Wrote to {output_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_name",
      type=str,
      help=(
          "The name of the model to evaluate. Currently support `gpt-4o`,"
          " `o1-preview`, `gemini-1.5-flash`, `gemini-1.5-pro`, `gemma_2b`,"
          " `gemma_27b`, and `gemma_9b`"
      ),
  )
  parser.add_argument(
      "--domain_name",
      type=str,
      choices=[
          "SL",
          "GSM_csp",
          "GSM_verbal",
          "Planning",
      ],
      help="Domain name",
  )
  parser.add_argument(
      "--eval_mode",
      type=str,
      choices=[
          "mc",
          "isambig",
          "fullinfo",
      ],
      help="Eval mode",
  )
  parser.add_argument("--data_file", type=str, help="Data file")
  parser.add_argument(
      "--data_dir",
      type=str,
      default="data",
      help="Directory containing data",
  )
  parser.add_argument(
      "--prompt_mode",
      type=str,
      choices=["", "cot", "fs4"],
      default="",
      help="Use vanilla, CoT, or fewshot prompting (with 4 samples)",
  )
  parser.add_argument(
      "--results_dir",
      type=str,
      default="results",
      help="Directory to write results to",
  )
  args = parser.parse_args()
  main(args)
