# QuestBench

Data and code for generating QuestBench data and evaluating LLMs on it.

## Installation
1. Begin by creating a conda environment to contain the packages needed for
QuestBench. You can install anaconda here: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
```bash
conda create -n questbench PYTHON=3.11
conda activate questbench
```

2. Install PyTorch following the instructions here: https://pytorch.org/get-started/locally/

3. Install the remaining requirements
```bash
pip install -r requirements.txt
```

## Download datasets
1. [Click here to download the datasets.](https://storage.googleapis.com/questbench/questbench_data.tar.gz)

2. After downloading, expand the compressed file.
```bash
tar -xzvf questbench_data.tar.gz
```

## Run evaluations
Set your api key to be able to use Gemini models
```bash
export GOOGLE_API_KEY=<gemini_api_key>
```

Login to HuggingFace to be able to use Gemma models
```bash
huggingface-cli login
```

Set your openai key to be able to use GPT models
```bash
export OPENAI_API_KEY=<openai_api_key>
export OPENAI_ORGANIZATION=<openi_organization_key>
export OPENAI_PROJECT=<openai_project_key>
```

Next, run the eval
```bash
python mc_eval.py \
--model_name [gemini-1.5-pro|gemini-1.5-flash|gpt-4o|o1-preview|gemma-27b|gemma-2b|gemma-9b] \
--domain_name [GSM_csp|Planning|SL|GSM_verbal] \
--eval_mode [mc|isambig|fullinfo] \
--data_dir <data_dir> \
--data_file <data_fp> \
--prompt_mode [|cot|fs4] \
--results_dir <results_dir>
```

* `--data_dir` should be set to the directory containing all the data files. By default, `--data_dir` is set to `questbench_data/`.
* `--data_file` should be set to the appropriate file for the domain. If you downloaded the datasets from the public website, the data files should be set to
```bash
questbench_data/Logic-Q/simplelogic_heldout_1k.csv
questbench_data/Planning-Q/planning_heldout_7500.csv
questbench_data/GSM-Q/gsm_CSP_heldout_pilot.csv
questbench_data/GSM-Q/gsm_verbal_heldout_pilot.csv
```

## Generate datasets
Before running any code, be sure to run
```bash
export PYTHONPATH=.
```

### Logic-Q
Generate 1-sufficient rulesets
```bash
python SimpleLogic/generate_ruleset.py \
      --sl_dir <sl_rules_dir> \
      --start_idx <start_idx> \
      --end_idx <end_idx>
```

Make Logic-Q data from 1-sufficient rulesets
```
python SimpleLogic/make_data.py \
      --sl_dir <sl_rules_dir> \
      --max_problems_to_sample_per_ruleset <max_problems_to_sample_per_ruleset>
```

### Planning-Q
Generate 1-sufficient CSPs
```bash
python Planning/make_planning_data.py \
      --pddl_dir <pddl_dir> \
      --output_dir <output_dir>
```

Run remaining commands in under "Make data" header in
Make Planning-Q data from 1-sufficient CSPs
```
python Planning/make_data.py \
      --input_dir <input_dir> \
      --output_dir <output_dir>
```
where `input_dir` is the `output_dir` from the previous command.

### GSM-Q
GSM-Q was created through human annotation.

Please see the technical report for more details.

## Citing this work
```
@techreport{li2025,
  title={QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks?},
  author={Belinda Li and Been Kim and Zi Wang},
  year={2025},
  institution={Google DeepMind}
}
```

## License and disclaimer
<!-- mdlint off(LINE_OVER_80) -->

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

