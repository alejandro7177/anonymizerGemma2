from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from utils import dataset
import torch

## CARGAMOS EL MODELO PARA FINETUNING

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

#print(f"tokenizer: {tokenizer.eos_token}, type: {type(tokenizer.eos_token)}")

model = FastLanguageModel.get_peft_model(
    model,
    r = 2, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # en GB


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,#dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,

        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 5,
        max_steps = 60,

        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "/home/data/nlp/anoni/gemma2-9b-prompt",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()
print('fin del entrenamiento :)')
# # @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

import matplotlib.pyplot as plt # type: ignore
import numpy as np

# Durante el entrenamiento
loss_history = []
steps = []

for step, loss in enumerate(trainer_stats.metrics['train_loss']):
    loss_history.append(loss)
    steps.append(step)

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_history, 'b-', label='Loss de entrenamiento')
plt.xlabel('Pasos')
plt.ylabel('Loss')
plt.title('Progreso del Entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig('progreso_entrenamiento.png')

## GUARDAR MODLEO

# model.save_pretrained("/home/data/nlp/anoni/gemma2-9b-finetunig-1-0")  # Local saving
# tokenizer.save_pretrained("/home/data/nlp/anoni/gemma2-9b-finetunig-1-0")