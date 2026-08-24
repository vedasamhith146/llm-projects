import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache,trim_prompt_cache
import time

target_model_path='mlx-community/Qwen2.5-1.5B-Instruct-4bit'
draft_model_path='mlx-community/Qwen2.5-0.5B-Instruct-4bit'
print("loading model...")

target_model,tokenizer=load(target_model_path)
draft_model, tokenizer=load(draft_model_path)

messages=[
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":"What is the capital of France?"}
]
prompt_set=[[{"role":"system","content":"You are a helpful and concise assistant."},
    {"role":"user","content":"What is the capital of France?"}],
    [{"role": "system", "content": "You are a creative writer."}, 
     {"role": "user", "content": "Write a 50-word story about a dragon who loves math."}],
    [[{"role": "system", "content": "You are a coding assistant."}, 
     {"role": "user", "content": "Write a Python function to reverse a linked list."}]],
    [{"role": "system", "content": "You are a helpful assistant."}, 
     {"role": "user", "content": "Explain quantum computing in simple terms in 3 sentences."}]
]

def next_token_generator(logits,temp=1,top_k=None,top_p=None):
    if temp==0:
        next_token=mx.argmax(logits,axis=-1,keepdims=True)
        probs=mx.softmax(logits,axis=-1)
        return next_token,probs
    logits=logits/temp

    if top_k is not None:
        topk_vals= mx.topk(logits,k=top_k,axis=-1)
        kth_value=mx.min(topk_vals,axis=-1,keepdims=True)
        logits=mx.where(logits<kth_value,mx.array(-float('inf')),logits)


    if top_p is not None:
        probs=mx.softmax(logits,axis=-1)
        sorted_indices=mx.argsort(-probs,axis=-1)
        sorted_probs=mx.take_along_axis(probs,sorted_indices,axis=-1) 
        cumm_prob=mx.cumsum(sorted_probs,axis=-1)
        sorted_remove=cumm_prob > top_p
        sorted_remove=mx.concatenate(
            [mx.zeros_like(sorted_remove[...,:1]),sorted_remove[...,:-1]],axis=-1)
        unsort_order=mx.argsort(sorted_indices,axis=-1)
        remove_mask=mx.take_along_axis(sorted_remove,unsort_order,axis=-1)
        logits=mx.where(remove_mask,mx.array(-float('inf')),logits)

    probs=mx.softmax(logits,axis=-1)
    next_token=mx.random.categorical(mx.log(probs))
    next_token=next_token[:,None]
    return next_token,probs


def generate_tokens(current_messages,num_tokens,gamma):

    gen_draft_model=0
    acc_draft_model=0

    #Converting prompt into mlx core array to process
    input_ids=tokenizer.apply_chat_template(current_messages,tokenize=True,add_generation_prompt=True,return_tensors="np")
    input_vector=mx.array(input_ids)

    #print(f"No of tokens in the initial prompt are {input_vector.shape[1]}")

    #Cache to store the keys,value of the prompt
    draft_cache=make_prompt_cache(draft_model)
    target_cache=make_prompt_cache(target_model)

    #initial prefill for draft and target models
    _=draft_model(input_vector,cache=draft_cache)
    target_logits=target_model(input_vector,cache=target_cache)
    next_token,_=next_token_generator(target_logits[:,-1,:])

   # print("Done with the initial prefilling for both the models")

    input_vector=mx.concatenate([input_vector,next_token],axis=1)
    corrected_tokens=1

    vocab_size=draft_model.args.vocab_size
    mx.eval(input_vector)
    start=time.perf_counter()

    total_beta=0.0
    total_positions=0

    while corrected_tokens<num_tokens:

        draft_token_probabilities=mx.zeros((0,vocab_size),dtype=mx.float32)
        draft_tokens=[]

        for _ in range(gamma):
            logits=draft_model(input_vector[...,-1:],cache=draft_cache)
            next_token,probs=next_token_generator(logits[:,-1,:])
            draft_token_probabilities=mx.concatenate([draft_token_probabilities,probs],axis=0)
            input_vector=mx.concatenate([input_vector,next_token],axis=1)
            draft_tokens.append(next_token)

        #print(f"Done with generating {gamma} tokens from the draft model")
    
        target_logits=target_model(input_vector[:,-gamma-1:],cache=target_cache)
        target_token_probabilities=mx.softmax(target_logits,axis=-1)


        all_accepted=True
        reached_limit=False
        for i in range(gamma):
            if corrected_tokens>=num_tokens:
                reached_limit=True
                break
            gen_draft_model+=1
            q_i=draft_token_probabilities[i]
            p_i=target_token_probabilities[0,i]
            beta_i = mx.sum(mx.minimum(p_i, q_i)).item()
            total_beta += beta_i
            total_positions += 1
            token=draft_tokens[i]
            ratio=min(1.0,(p_i[token]/q_i[token]).item())
            r=mx.random.uniform().item()
            if r<=ratio:
                corrected_tokens+=1
                acc_draft_model+=1
                continue   
            else:
                input_vector=input_vector[:,:-gamma+i]
                trim_prompt_cache(draft_cache,gamma-1-i)
                trim_prompt_cache(target_cache,gamma-i)
                all_accepted=False
                break


        if not reached_limit:

            if all_accepted:
                next_token,_=next_token_generator(target_logits[:,-1,:])
                input_vector=mx.concatenate([input_vector,next_token],axis=1)
                _ = draft_model(input_vector[:, -gamma-1:-gamma],cache=draft_cache)
            else:
                curr_distribution=mx.expand_dims((target_token_probabilities[0,i]-draft_token_probabilities[i]),axis=0)
                curr_distribution=mx.where(curr_distribution<0,mx.array(float('0')),curr_distribution)
                curr_distribution=curr_distribution/mx.sum(curr_distribution,axis=-1,keepdims=True)
                next_token=mx.random.categorical(mx.log(curr_distribution))
                next_token=next_token[:,None]
                input_vector=mx.concatenate([input_vector,next_token],axis=1)

            corrected_tokens+=1


    

        #print(f"corrected_tokens={corrected_tokens}")
    mx.eval(input_vector)
    end=time.perf_counter()
    alpha_local=total_beta/total_positions if total_positions>0 else 0
    time_taken=end-start
    return total_beta,total_positions,alpha_local,time_taken
if __name__=='__main__':
    total_beta_global=0
    total_positions_global=0

    print("Measuring alpha")
    for i,current_messages in enumerate(prompt_set):
        print(f"Running prompt {i+1}/{len(prompt_set)}")
        beta_sum,positions,alpha_local,_=generate_tokens(current_messages,num_tokens=100,gamma=5)
        total_beta_global+=beta_sum
        total_positions_global+=positions
        print(f" -> Local alpha: {alpha_local:.4f} (based on {positions} draft positions)")

    if total_positions_global>0:
        alpha_measured=total_beta_global/total_positions_global
        print(f"\n Accurate global alpha over {len(prompt_set)}")
        print(f"total draft positions:{total_positions_global}")
        print(f"alpha={alpha_measured:4f}")
    else:
        print("Error: No positions measured.")









