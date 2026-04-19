# !pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/functiongemma-270m-it",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add Adapters (LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
)

# 3. Load Data (Upload training_data.jsonl to Colab first)
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Crucial: BOS/EOS and Developer role
        text = (
            f"{tokenizer.bos_token}<start_of_turn>developer\n{instruction}<end_of_turn>\n"
            f"<start_of_turn>user\n{input_text}<end_of_turn>\n"
            f"<start_of_turn>model\n{output}<end_of_turn>{tokenizer.eos_token}"
        )
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 450,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "outputs",
    ),
)

# 5. Train
trainer.train()

# 6. Save as GGUF
model.save_pretrained_gguf("function_gemma_router", tokenizer, quantization_method = "q8_0")