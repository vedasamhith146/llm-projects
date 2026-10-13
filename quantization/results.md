# Qwen2.5-1.5B-Instruct Quantization Results

| Model | Format | Size | Memory (RSS) | Prompt Processing (pp512) | Generation (tg128) | Perplexity |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | F16 GGUF | 2.88 GiB | 2.54 GiB | 1064.31 tok/s | 26.40 tok/s | 8.6659 |
| Qwen2.5-1.5B-Instruct | Q8_0 GGUF | 1.53 GiB| 2.46 GiB | 982.76 tok/s | 51.16 tok/s | 8.6691  |
| Qwen2.5-1.5B-Instruct | Q4_0 GGUF | 885.97 MiB | 1.81 GiB | 1036.35 tok/s  | 85.17 tok/s | 9.4665 |
| Qwen2.5-1.5B-Instruct | Q4_K_M GGUF| 934.69 MiB | 1.84 GiB | 979.98 tok/s | 78.39 tok/s | 9.0455 |