"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True
    )

    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
   
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
import bitsandbytes as bnb

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    
    
    return any(
        isinstance(module, bnb.nn.Linear4bit)
        for module in model.modules()

    )

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    return target_modules

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return FastLanguageModel.get_peft_model(
        model,
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules

    )

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
   trainable_fraction = trainable_count / total_count
   return(trainable_fraction)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    
    return [
        {
            "instruction": "What is machine learning?",
            "response": "Machine learning is a field of AI where models learn patterns from data."
        },
        {
            "instruction": "Translate 'Hello' to French.",
            "response": "Bonjour"
        },
        {
            "instruction": "Summarize the benefits of exercise.",
            "response": "Regular exercise improves cardiovascular health, strength, and mental well-being."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    )

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids[:])

# Step 15 - build_training_arguments
def build_training_arguments(output_dir="./sft_out", max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
    )

# Step 16 - build_sft_trainer
from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments


def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl.SFTTrainer over dataset['text'] ready to .train()."""

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    return trainer

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""

    # Disable features that may trigger multiprocessing/checkpointing
    trainer.args.dataloader_num_workers = 0
    trainer.args.report_to = []
    trainer.args.save_strategy = "no"

    train_result = trainer.train()
    return float(train_result.training_loss)

# Step 18 - switch_to_inference_mode
from unsloth import FastLanguageModel

def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    FastLanguageModel.for_inference(model)
    return model

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

