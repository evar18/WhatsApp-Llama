# Table of Contents
1. [About "You"](#quick-start)
2. [Results](#results)
3. [Setup](#fine-tuning)
    - [Get WhatsApp Chats](#single-gpu)
    - [Create Dataset](#multiple-gpus-one-node)
    - [Finetune](#multi-gpu-multi-node)
4. [Future Work](./docs/inference.md)
5. [License and Acceptable Use Policy](#license)


# About "You"

Finetune a Llama 7B Chat model on your WhatsApp conversations, and teach it to reply like you! Llama 7B chat is finetuned using parameter efficient finetuning (QLoRA) and int4 quantization on a single GPU (P100 with 16GB gpu memory). The entire experiment can be run for free using the $300 of Google Cloud compute credits received on new user sign ups.

# Results
LLama7B learned my texting style extremely quickly. Here are the changes that occurred post finetuning:
1. Average words generated in output went down by x%
2. it learnt my emoji usage
3. It picked up on common phrases I use (Whaddup, 1 more)

I personally ran a "Turing test" on my friends, where each of them asked me 3 questions on WhatsApp, and both I and finetuned LLaMa answered each question. Finetuned Llama7B managed to convince 3 of my x friends that it was the real Advaith, which is astounding given its extremely limited training period.

# Setup

[Llama 2 Jupyter Notebook](quickstart.ipynb): This jupyter notebook steps you through how to finetune a Llama 2 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum). 

**Note** All the setting defined in [config files](./configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.


## Requirements
To run the examples, make sure to install the requirements using

```bash
# python 3.9 or higher recommended
pip install -r requirements.txt

```

**Please note that the above requirements.txt will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

# Where to find the models?

You can find llama v2 models on HuggingFace hub [here](https://huggingface.co/meta-llama), where models with `hf` in the name are already converted to HuggingFace checkpoints so no further conversion is needed. The conversion step below is only for original model weights from Meta that are hosted on HuggingFace model hub as well.

# Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Llama 2 model definition provided by Hugging Face's transformers library.

Given that the original checkpoint resides under models/7B you can install all requirements and convert the checkpoint with:

```bash
## Install HuggingFace Transformers from source
pip freeze | grep transformers ## verify it is version 4.31.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

# Fine-tuning

For fine-tuning Llama 2 models for your domain-specific use cases recipes for PEFT, FSDP, PEFT+FSDP have been included along with a few test datasets. For details see [LLM Fine-tuning](./docs/LLM_finetuning.md).

## Single and Multi GPU Finetune

If you want to dive right into single or multi GPU fine-tuning, run the examples below on a single GPU like A10, T4, V100, A100 etc.
All the parameters in the examples and recipes below need to be further tuned to have desired results based on the model, method, data and task at hand.

**Note:**
* To change the dataset in the commands below pass the `dataset` arg. Current options for dataset are `grammar_dataset`, `alpaca_dataset`and  `samsum_dataset`. A description of the datasets and how to add custom datasets can be found in [Dataset.md](./docs/Dataset.md). For  `grammar_dataset`, `alpaca_dataset` please make sure you use the suggested instructions from [here](./docs/single_gpu.md#how-to-run-with-different-datasets) to set them up.

* Default dataset and other LORA config has been set to `samsum_dataset`.

* Make sure to set the right path to the model in the [training config](./configs/training.py).

### Single GPU:

```bash
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

Here we make use of Parameter Efficient Methods (PEFT) as described in the next section. To run the command above make sure to pass the `peft_method` arg which can be set to `lora`, `llama_adapter` or `prefix`.

**Note** if you are running on a machine with multiple GPUs please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`

**Make sure you set [save_model](configs/training.py) in [training.py](configs/training.py) to save the model. Be sure to check the other training settings in [train config](configs/training.py) as well as others in the config folder as needed or they can be passed as args to the training script as well.**


### Multiple GPUs One Node:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that int8 quantization from bit&bytes currently is not supported in FSDP.

```bash

torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model

```

Here we use FSDP as discussed in the next section which can be used along with PEFT methods. To make use of PEFT methods with FSDP make sure to pass `use_peft` and `peft_method` args along with `enable_fsdp`. Here we are using `BF16` for training.

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up the fine-tuning job. This has been enabled in `optimum` library from HuggingFace as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model --use_fast_kernels
```

### Fine-tuning using FSDP Only

If you are interested in running full parameter fine-tuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --use_fast_kernels

```

### Fine-tuning using FSDP on 70B Model

If you are interested in running full parameter fine-tuning on the 70B model, you can enable `low_cpu_fsdp` mode as the following command. This option will load model on rank0 only before moving model to devices to construct FSDP. This can dramatically save cpu memory when loading large models like 70B (on a 8-gpu node, this reduces cpu memory from 2+T to 280G for 70B model). This has been tested with `BF16` on 16xA100, 80GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8 llama_finetuning.py --enable_fsdp --low_cpu_fsdp --pure_bf16 --model_name /patht_of_model_folder/70B --batch_size_training 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

### Multi GPU Multi Node:

```bash

sbatch multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```
You can read more about our fine-tuning strategies [here](./docs/LLM_finetuning.md).


# Repository Organization
This repository is organized in the following way:

[configs](configs/): Contains the configuration files for PEFT methods, FSDP, Datasets.

[docs](docs/): Example recipes for single and multi-gpu fine-tuning recipes.

[ft_datasets](ft_datasets/): Contains individual scripts for each dataset to download and process. Note: Use of any of the datasets should be in compliance with the dataset's underlying licenses (including but not limited to non-commercial uses)


[inference](inference/): Includes examples for inference for the fine-tuned models and how to use them safely.

[model_checkpointing](model_checkpointing/): Contains FSDP checkpoint handlers.

[policies](policies/): Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode).

[utils](utils/): Utility files for:

- `train_utils.py` provides training/eval loop and more train utils.

- `dataset_utils.py` to get preprocessed datasets.

- `config_utils.py` to override the configs received from CLI.

- `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.

- `memory_utils.py` context manager to track different memory stats in train loop.

# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)
