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
    {"role":"system","content":"You are an helpful assistant."},
    {"role":"user","content":"You are a creative storyteller. Write a detailed, 150-word story about a robot who discovers a hidden garden on a deserted planet. Describe the sights, sounds, and emotions."}
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


def generate_tokens_speculative(messages,num_tokens,gamma):

    total_beta=0.0
    total_positions=0

    #Converting prompt into mlx core array to process
    input_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="np")
    input_vector=mx.array(input_ids)

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
            q_i=draft_token_probabilities[i]
            p_i=target_token_probabilities[0,i]
            beta_i=mx.sum(mx.minimum(p_i,q_i)).item()
            total_beta+=beta_i
            total_positions+=1
            token=draft_tokens[i]
            ratio=min(1.0,(p_i[token]/q_i[token]).item())
            r=mx.random.uniform().item()
            if r<=ratio:
                corrected_tokens+=1
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
                #_ = draft_model(input_vector[:, -gamma-1:-gamma],cache=draft_cache)
            else:
                curr_distribution=mx.expand_dims((target_token_probabilities[0,i]-draft_token_probabilities[i]),axis=0)
                curr_distribution=mx.where(curr_distribution<0,mx.array(float('0')),curr_distribution)
                curr_distribution=curr_distribution/mx.sum(curr_distribution,axis=-1,keepdims=True)
                next_token=mx.random.categorical(mx.log(curr_distribution))
                next_token=next_token[:,None]
                input_vector=mx.concatenate([input_vector,next_token],axis=1)

            corrected_tokens+=1

    mx.eval(input_vector)
    end=time.perf_counter()
    time_taken=end-start

    alpha_local=total_beta/total_positions if total_positions>0 else 0
    return time_taken,alpha_local

def generate_tokens_target(messages,num_tokens):

    #Converting prompt into mlx core array to process
    input_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="np")
    input_vector=mx.array(input_ids)

   
    #Cache to store the keys,value of the prompt
    target_cache=make_prompt_cache(target_model)

    #initial prefill for draft and target models
    target_logits=target_model(input_vector,cache=target_cache)
    next_token,_=next_token_generator(target_logits[:,-1,:])
    input_vector=mx.concatenate([input_vector,next_token],axis=1)
    corrected_tokens=1

    mx.eval(input_vector)
    start=time.perf_counter()
    while corrected_tokens<num_tokens:
        logits=target_model(input_vector[...,-1:],cache=target_cache)
        next_token,probs=next_token_generator(logits[:,-1,:])
        input_vector=mx.concatenate([input_vector,next_token],axis=1)
        corrected_tokens+=1
    mx.eval(input_vector)
    end=time.perf_counter()

    return end-start

if __name__ == '__main__':
    num_tokens = 150
    gamma = 1
    runs = 5
    
    total_time_spec = 0
    total_time_target = 0
    alpha_sum = 0.0
    
    for i in range(runs):
        print(f"Run {i+1}/{runs}...")
        
        # Speculative
        time_spec, alpha = generate_tokens_speculative(messages, num_tokens, gamma)
        total_time_spec += time_spec
        alpha_sum += alpha
        
        # Target only
        time_target = generate_tokens_target(messages, num_tokens)
        total_time_target += time_target
    
    avg_time_spec = total_time_spec / runs
    avg_time_target = total_time_target / runs
    avg_alpha = alpha_sum / runs
    
    speedup = avg_time_target / avg_time_spec  # Note: target/spec = speedup
    print(f"Target average time: {avg_time_target:.4f}s")
    print(f"Speculative average time: {avg_time_spec:.4f}s")
    print(f"Speedup: {speedup:.4f}x")
    print(f"Measured α: {avg_alpha:.4f}")