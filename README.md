#  Custom LLM Architecture & Pre-training Engine

Welcome to my active research repository. This workspace contains a complete, from-scratch implementation of modern Large Language Model architectures, optimizations, and an end-to-end 124M parameter pre-training pipeline.

 **Read my deep-dive technical articles on Medium:** https://medium.com/@githubveda

---

## 1.  Large Language Model Pre-training (GPT-2 124M)
**Folders:** `p04_gpt2_from_scratch`, `p11_pretraining_objectives`

An end-to-end pipeline for pre-training a custom GPT-2 model. 
* **Dataset:** FineWeb-Edu
* **Hardware:** 1x NVIDIA RTX 5090 (32GB VRAM) via RunPod
* **Milestone:** Successfully trained for 11.5k steps (~0.6 epochs) before hitting compute limits. Model successfully generates coherent text.
* **Current Focus:** Architecting Supervised Fine-Tuning (SFT) and Reinforcement Learning from Verifiable Rewards (RLVR) pipelines.

*> Note: Full training loss logs were lost due to an ephemeral cloud GPU instance termination. However, model checkpoints up to step 11.5k (~0.6 epochs) were successfully salvaged. While the model has not yet reached full convergence due to compute budget limits, early sample generations demonstrate it is successfully learning basic English syntax, vocabulary, and structure.*

<img width="1378" height="169" alt="Screenshot 2026-03-27 at 6 55 20 PM" src="https://github.com/user-attachments/assets/8f12f971-d0d3-410a-8fce-5e9b53b97327" />
<img width="1373" height="145" alt="Screenshot 2026-03-27 at 7 01 55 PM" src="https://github.com/user-attachments/assets/c620ebc3-c925-4470-8684-e0ff96db4371" />
<img width="1363" height="149" alt="Screenshot 2026-03-27 at 7 03 32 PM" src="https://github.com/user-attachments/assets/caec0052-8b48-4d4b-8dc2-00623b24b94c" />
<img width="1371" height="111" alt="Screenshot 2026-03-27 at 7 04 49 PM" src="https://github.com/user-attachments/assets/c64ba835-b03a-443c-96fb-4f676901143a" />


---

## 2.  Advanced Architecture & Optimizations
**Folders:** `p01` to `p03`, `p04_transformers_qkv_stacking`, `p08_moe`, `p09_grouped_query_attention`, `p10_normalization_activations`

A comprehensive framework building Transformer components from the ground up to understand mathematical mechanics and hardware efficiency.
* **Core:** Byte-Pair Encoding (BPE), Causal Multi-head Attention, SwiGLU, RMSNorm.
* **Efficiency:** Implemented a dynamic 2-expert Mixture of Experts (MoE) routing layer and Grouped Query Attention (GQA).
* **Ablation Studies:** Conducted rigorous ablations on positional embeddings (RoPE, ALiBi, Sinusoidal) to visualize attention collapse.

*<img width="998" height="600" alt="Screenshot 2026-03-27 at 7 13 11 PM" src="https://github.com/user-attachments/assets/51f51c1f-0ec2-40f5-a436-31c00700bd0d" />
*
*(Above: Expert Utilization histogram for the custom Top-2 MoE routing layer, demonstrating successful load balancing and avoidance of routing collapse across 8 simulated experts).*

---

## 3. Inference, Sampling & Caching
**Folders:** `p05_sampling_topk_topp`, `p06_kv_cache`

Tools and optimizations for text generation and inference latency.
* **Interactive Dashboard:** Built a Streamlit application to visualize the mathematical impact of temperature, top-k, and top-p sampling, plotting entropy vs. output diversity.
* **KV Cache:** Custom implementation to measure $O(N)$ memory growth and inference speedups.

*(Drag and drop a screenshot of your Streamlit dashboard or entropy plot here)*
