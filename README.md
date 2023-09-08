# WhatsApp-Llama-Model: Fine-tune LLM to Mimic Your WhatsApp Style!

This repository is a fork of the `facebookresearch/llama-recipes`, adapted to fine-tune a Llama 7b chat model to replicate your personal WhatsApp texting style. By simply inputting your WhatsApp conversations, you can train the LLM to respond just like you do! Llama 7B chat is finetuned using parameter efficient finetuning (QLoRA) and int4 quantization on a single GPU (P100 with 16GB gpu memory).

## My Results

1. **Quick Learning**: The fine-tuned Llama model picked up on my texting nuances rapidly. The average words generated in the output reduced by x%, and the model accurately replicated my unique emoji usage and my common phrases, like "Whaddup" and "1 more".

2. **Turing Test with Friends**: As an experiment, I had the LLM chat with some of my friends without them knowing. The result? It fooled 2 out of 20 of them! Some of the model's responses were eerily similar to my own. Here are some examples:
    - *Example 1*: [Add text here]
    - *Example 2*: [Add text here]
    - *Example 3*: [Add text here]

## Getting Started

Here's a step-by-step guide on setting up this repository and creating your own customized dataset:

### 1. Exporting WhatsApp Chats
[Provide detailed instructions on how to export WhatsApp chats, e.g., selecting a chat -> clicking on the three dots at the top right -> More -> Export chat]

### 2. Preprocessing the Dataset
Run the provided preprocessing files to convert the exported chat into a format suitable for training:

```bash
python preprocessing_script.py --input_path=<path_to_exported_chat> --output_path=<path_to_preprocessed_data>
```

### 3. Validating the Dataset
Here's the expected format for the preprocessed dataset:

```
[Provide example format here]
```

Ensure your dataset looks like the above to verify you've done it correctly.

### 4. Model Configuration
- If you're using a **P100 GPU**, load the model in **4 bits**:
    ```bash
    python train_script.py --bit_precision=4 --gpu_type=P100
    ```

- If you're using an **A100 GPU**, load the model in **8 bits**:
    ```bash
    python train_script.py --bit_precision=8 --gpu_type=A100
    ```

### 5. Training Time
For reference, a 10MB dataset will complete 1 epoch in approximately 7 hours on a P100 GPU. My results shared above were achieved after training for just 1 epoch.

## Conclusion

This adaptation of the Llama model offers a fun way to see how well a LLM can mimic your personal texting style. Remember to use AI responsibly and inform your friends if you're using the model to chat with them.

---

Happy training, and have fun watching LLM text like you!



