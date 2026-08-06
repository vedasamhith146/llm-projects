import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments
from datasets import load_dataset

device='mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device {device}")

model_name="gpt2"
tokenizer=AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
model.to(device)

lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn","c_proj","c_fc"],
    task_type=TaskType.CAUSAL_LM
)

model=get_peft_model(model,lora_config)

dataset=load_dataset("tatsu-lab/alpaca", split="train")

def format_short_template(example):
    instruction = example['instruction']
    input_text = example.get('input', '') 
    output = example['output']
    
    # Build the base text
    if input_text.strip():
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    # Always append the end-of-text token
    text += tokenizer.eos_token
    
    return {"text": text}

dataset = dataset.map(format_short_template)
dataset=dataset.filter(lambda x: len(tokenizer(x["text"],truncation=True)["input_ids"])<=512)
print(f"Final dataset size is : {len(dataset)} examples")

training_args=TrainingArguments(
    output_dir="./gpt2-alpaca-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

trainer=SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    #max_length=512,
    #dataset_text_field="text",
    #packing=False
)

print("Starting training....")
trainer.train()

trainer.save_model("./gpt2-alpaca-lora-final")
print("Training completed and model saved")

print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()
torch.save(merged_model.state_dict(), "gpt2_instruction_tuned.pt")
print("Saved single .pt file: gpt2_instruction_tuned.pt")


