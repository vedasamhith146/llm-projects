import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache,trim_prompt_cache
import time

target_model_path='mlx-community/Qwen2.5-1.5B-Instruct-4bit'

print("loading model...")

target_model,tokenizer=load(target_model_path)

messages=[
    {"role":"system","content":"You are a helpful and concise assistant."},
    {"role":"user","content":"Write a detailed science-fiction story about a spacecraft that lands on a mysterious planet after losing communication with Earth. Describe the alien landscape, the spacecraft interior, the sounds and atmosphere around the crew, and the discovery that something unexpected is moving beneath the surface. Build the story gradually and end with a surprising discovery."}
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


def generate_tokens_target(num_tokens):

    #Converting prompt into mlx core array to process
    input_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="np")
    input_vector=mx.array(input_ids)

   
    #Cache to store the keys,value of the prompt
    target_cache=make_prompt_cache(target_model)

    #initial prefill for target model
    target_logits = target_model(input_vector, cache=target_cache)
    next_token, _ = next_token_generator(target_logits[:, -1, :])
    
    mx.eval(next_token)

    start=time.perf_counter()

    for _ in range(num_tokens):
        logits=target_model(next_token,cache=target_cache)
        next_token,_=next_token_generator(logits[:,-1,:])
    
    mx.eval(next_token)

    end=time.perf_counter()

    return end-start

if __name__=='__main__':
    num_tokens=150

    for _ in range(10):
        generate_tokens_target(num_tokens)

    times=[]
    for i in range(30):
        time_taken=generate_tokens_target(num_tokens)
        times.append(time_taken)
    
    print(f"Target Model token/sec : {num_tokens/(sum(times)/len(times))} tok/s")
    print(f"Target Model latency/token : {((sum(times)/len(times))/num_tokens)*1000} ms/tok")
   










