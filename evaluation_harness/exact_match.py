import json
import sqlite3

prompts = []

with open("eval_dataset.jsonl") as f:
    for line in f:
        prompts.append(json.loads(line))

connection = sqlite3.connect("eval.db")
cursor = connection.cursor()

cursor.execute("""
    SELECT raw_output
    FROM generations
    WHERE category IN (?, ?)
""", ("safe_factual", "grounded_qa"))

rows = cursor.fetchall()

exact_matches = 0
total_evaluations = 0

with open("comparing.jsonl", "w") as f:

    #prompts from 60->90 are factual QA (safe_factual,grounded_qa)
    
    for i in range(60, 90):

        correct_answer = (
            prompts[i]["ground_truth"]
            .lower()
            .replace(" ", "")
        )


        start = (i - 60) * 9
        end = start + 9

        for row in rows[start:end]:

            provided_answer = (
                row[0]
                .lower()
                .replace(" ", "")
            )

            total_evaluations += 1

            comparison = {
                "correct_answer": correct_answer,
                "provided_answer": provided_answer
            }

            f.write(json.dumps(comparison) + "\n")

            if correct_answer == provided_answer:
                exact_matches += 1


exact_match_rate = exact_matches / total_evaluations

print(f"Exact-match rate: {exact_match_rate:.4f}")



    


    
    
