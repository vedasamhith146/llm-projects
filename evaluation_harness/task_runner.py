import json,sqlite3
import time
import requests
import sys
import math

prompts=[]
with open("eval_dataset.jsonl","r") as f:
    for line in f:
        prompt=json.loads(line)
        prompts.append(prompt)

connection=sqlite3.connect('eval.db')

cursor=connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS generations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER,
    category TEXT,
    template_key TEXT,
    seed INTEGER,
    temperature REAL,
    system_prompt TEXT,
    user_prompt TEXT,
    raw_output TEXT,
    first_token_logprob REAL,
    top_5_tokens TEXT,
    latency_seconds REAL,
    total_tokens INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prompt_id,template_key,seed))
""")

connection.commit()

system_prompts={"sys_default":"You are a helpful AI assistant.","sys_expert":"You are a senior domain expert.Provide detailed, technical, and precise responses.","sys_urgent":"You are a helpful assistant.The user is under time pressure and needs a direct,concise answer."}
seeds_and_temps=[(42,0.0),(123,0.7),(999,1.0)]

generations_before_commit=0


for prompt in prompts:
    for template_key in prompt["prompt_templates"]:
        for (seed,temp) in seeds_and_temps:
            #if already there then continue
            user_prompt=prompt['prompt_templates'][template_key]
            system_prompt=system_prompts["sys_default"]
            prompt_id=prompt['id']
            category=prompt['category']

            cursor.execute("""
                SELECT 1
                FROM generations
                WHERE prompt_id = ?
                AND template_key = ?
                AND seed = ?
            """, (prompt_id, template_key, seed))

            if cursor.fetchone() is not None:
                print(f"Skipping: prompt {prompt_id}, {template_key}, seed {seed}")
                continue


            start=time.time()
            response=requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":"phi3.5:3.8b",
                    "system":system_prompt,
                    "prompt":user_prompt,
                    "stream": False,
                    "logprobs":True,
                    "top_logprobs":5,
                    "options":{
                        "seed":seed,
                        "temperature":temp,
                        "num_predict":256
                    }


                }, 
                timeout=30
            )
            data=response.json()
            raw_output=data['response']
            total_tokens=data["prompt_eval_count"]+data["eval_count"]
            latency=data["total_duration"]/10**9
            first_token_logprob=data["logprobs"][0]["logprob"]
            top_5_tokens=json.dumps(
                data["logprobs"][0]["top_logprobs"]
            )

            cursor.execute("""
            INSERT INTO generations(
                prompt_id,
                category,
                template_key,
                seed,
                temperature,
                system_prompt,
                user_prompt,
                raw_output,
                first_token_logprob,
                top_5_tokens,
                latency_seconds,
                total_tokens
                )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,(
                prompt_id,
                category,
                template_key,
                seed,
                temp,
                system_prompt,
                user_prompt,
                raw_output,
                first_token_logprob,
                top_5_tokens,
                latency,
                total_tokens
            ))

            generations_before_commit+=1
            print(f"Inserted: prompt {prompt_id}, {template_key}, seed {seed}")
            if generations_before_commit>=10:
                connection.commit()
                generations_before_commit=0



connection.commit()        









