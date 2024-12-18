# QuestBench

Data and code for generating QuestBench data and evaluating LLMs on it.

## Installation

```bash
conda create -n questbench PYTHON=3.11
conda activate questbench
pip install -r requirements.txt
```


## Generate datasets
Before running any code, be sure to run
```bash
export PYTHONPATH=.
```

### Logic-Q
```bash
python SimpleLogic/generate_ruleset.py \
      --start_idx <start_idx> \
      --end_idx <end_idx>
```

Run remaining commands in under "Make data" header in
```
SimpleLogic/make_data.ipynb
```

### Planning-Q
```bash
python Planning/scripts/make_planning_data.py
```

Run remaining commands in under "Make data" header in
```
pyperplan/make_data.ipynb
```

## Run experiments
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
      --model_name [gemini_mpp_32k|gemini_flash_s_32k|gpt-4o|o1-preview|gemma-27b|gemma-2b|gemma-9b] \
      --domain_name [GSM_csp|plan|SL|GSM_verbal] \
      --eval_mode [mc|isambig|fullinfo] \
      --data_file <data_fp> \
      --prompt_mode [|cot|fs4]
```

Data file paths:
datasets/gsm_csp_heldout_570.csv
datasets/simplelogic_heldout_1k_o1_subsample.csv
datasets/simplelogic_heldout_1k.csv
datasets/planning_heldout_7500.csv
datasets/planning_heldout_7500_o1_subsample.csv

New GSM datasets
datasets/gsm_CSP_heldout_pilot.csv
datasets/gsm_verbal_heldout_pilot.csv



## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@article{publicationname,
      title={Publication Name},
      author={Author One and Author Two and Author Three},
      year={2024},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
