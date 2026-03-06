from datasets import load_dataset

dataset=load_dataset("tatsu-lab/alpaca")

subset=dataset["train"].select(range(2000))

with open("finetune.txt","w",encoding='utf-8') as f:
    for example in subset:
        instruction=example["instruction"]
        input_text=example["input"]
        output=example["output"]

        if input_text.strip() !="":
            text=f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}\n\n"
        else:
            text=f"Instruction:{instruction}\nResponse:{output}\n\n"

        f.write(text)

print("Saved 2000 examples to finetune.txt")