import torch
import torch.nn.functional as F
from model_kvcache import GPT2, GPT2Config
import tiktoken
import re

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint = "gpt2_instruction_tuned_transposed.pt" # Change to test base model!

# Load model
state_dict = torch.load(checkpoint, map_location=device)
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model = GPT2(GPT2Config(vocab_size=50257))
model.to(device)
model.load_state_dict(state_dict)
model.eval()

enc = tiktoken.get_encoding('gpt2')
EOS_TOKEN_ID = 50256

# --- THE 5 STRICT TESTS ---
# Each test has a prompt and a specific Python function to check if the rule was followed.
eval_tests = [
    {
        "name": "Sentence Count Constraint",
        "prompt": "Write a response that is exactly three sentences long.",
        # Check: Count periods, exclamation marks, or question marks.
        "check": lambda text: len(re.findall(r'[.!?]+', text)) == 3
    },
    {
        "name": "Mandatory Keyword",
        "prompt": "Explain what a 'computer' is. You MUST include the word 'silicon' in your response.",
        # Check: Is the word 'silicon' in the text?
        "check": lambda text: 'silicon' in text.lower()
    },
    {
        "name": "Negative Constraint",
        "prompt": "Describe the color red. Do not use the word 'blood' anywhere in your response.",
        # Check: Is the word 'blood' absent?
        "check": lambda text: 'blood' not in text.lower()
    },
    {
        "name": "Formatting Constraint",
        "prompt": "List the numbers 1, 2, and 3. Format your response as a bulleted list using hyphens (-).",
        # Check: Are there at least 3 hyphens?
        "check": lambda text: text.count('-') >= 3
    },
    {
        "name": "JSON-like Formatting",
        "prompt": "Give me the capital of France. Format your answer exactly like this: {'city': 'Paris'}",
        # Check: Does it contain curly braces and the word 'city'?
        "check": lambda text: '{' in text and '}' in text and 'city' in text.lower()
    }
]

def generate_eval(prompt, max_new_tokens=100):
    model.clear_kv_cache()
    initial_tokens = enc.encode(prompt)
    tokens = torch.tensor(initial_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = set(initial_tokens)

    with torch.no_grad():
        logits, _ = model(tokens, start_pos=0)
        logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            # Repetition penalty
            for token_id in generated_tokens:
                logits[0, token_id] = logits[0, token_id] / 1.2 if logits[0, token_id] > 0 else logits[0, token_id] * 1.2

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == EOS_TOKEN_ID:
                break

            generated_tokens.add(next_token.item())
            tokens = torch.cat((tokens, next_token), dim=-1)
            logits, _ = model(next_token, start_pos=tokens.size(1) - 1)
            logits = logits[:, -1, :]

    full_text = enc.decode(tokens.squeeze(0).tolist())
    # Strip the prompt to get ONLY the generated response
    return full_text.replace(prompt, "").strip()

print("Starting Strict Instruction Evaluation...")
print("="*60)

passed = 0

for i, test in enumerate(eval_tests, 1):
    print(f"\nTest {i}: {test['name']}")
    print(f"Prompt: {test['prompt']}")
    
    formatted_prompt = f"### Instruction:\n{test['prompt']}\n\n### Response:\n"
    response = generate_eval(formatted_prompt, max_new_tokens=100)
    
    print(f"Model Output: {response}")
    
    # Run the specific check for this test
    is_correct = test['check'](response)
    
    if is_correct:
        print("✅ PASS")
        passed += 1
    else:
        print("❌ FAIL")

accuracy = (passed / len(eval_tests)) * 100
print("\n" + "="*60)
print(f"FINAL INSTRUCTION FOLLOWING SCORE: {accuracy:.1f}% ({passed}/{len(eval_tests)})")
print("="*60)