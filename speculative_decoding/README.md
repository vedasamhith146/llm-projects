# Speculative Decoding on Apple Silicon — Experimental Results

This repository records the current experiments for speculative decoding using MLX on an Apple M3 MacBook Air. It documents the model pair, benchmark setup, measured results, observations, and the next steps toward a consolidated experimental harness.

The purpose of this README is to preserve the results obtained so far and explain what they indicate. The tables below reproduce the current experiment outputs; measurements from separate scripts or experiments are kept separate where their reported baselines differ.

---

## 1. Experimental setup

### Hardware

| Component | Configuration |
|---|---|
| Machine | MacBook Air, Apple M3 |
| Unified memory | 8 GB |
| Acceleration | MLX / Apple Metal |
| Operating system | macOS |

### Models

| Role | Model |
|---|---|
| Target model | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Draft model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |

### Speculative benchmark configuration

| Parameter | Value |
|---|---:|
| Generated tokens | 150 |
| Timed tokens | 149 |
| Temperature | 1.0 |
| Top-k | `None` |
| Top-p | `None` |
| Warmup runs | 10 |
| Measurement runs | 30 |
| Gamma sweep | `1, 2, 4, 6, 8, 12, 16` |

The target-only and speculative benchmarks are intended to use the same prompt, generation length, sampling settings, hardware, and run counts. Where separate experiment outputs report different target-only baselines, they are documented separately rather than silently combined.

---

## 2. Target-only baseline

The standalone `target_only.py` run reported:

| Metric | Result |
|---|---:|
| Throughput | 77.33314739037071 tok/s |
| Latency per token | 12.931065574663483 ms/token |

Throughput and latency are reciprocal measures when calculated over the same timed token count and interval:

\[
\text{Latency (ms/token)} \approx \frac{1000}{\text{Throughput (tok/s)}}
\]

The reported values are consistent with this relationship.

---

## 3. Speculative benchmark — gamma = 1

The gamma-1 speculative benchmark reported:

| Metric | Result |
|---|---:|
| Generated tokens | 150 |
| Timed tokens | 149 |
| Gamma | 1 |
| Temperature | 1.0 |
| Top-k | `None` |
| Top-p | `None` |
| Warmup runs | 10 |
| Measurement runs | 30 |
| Average time | 1.8749 s |
| Standard deviation of time | 0.0860 s |
| Throughput | 79.47 tok/s |
| Latency | 12.583 ms/token |
| Expected alpha | 0.5933 |
| Realized alpha | 0.5957 |
| Accepted draft tokens | 1,670 |
| Proposed draft tokens | 2,815 |

The realized alpha is the arithmetic mean of the per-run acceptance ratios:

\[
\bar{\alpha}=\frac{1}{R}\sum_{i=1}^{R}\alpha_i
\]

where \(R=30\). This mean need not equal the aggregate acceptance ratio calculated by dividing total accepted tokens by total proposed tokens.

Using the reported totals, the aggregate ratio is:

\[
\alpha_{\text{aggregate}}=\frac{1670}{2815}\approx0.5932
\]

Both values can differ because the mean per-run ratio weights every run equally, whereas the aggregate ratio gives more weight to runs with more proposed tokens.

### Standalone comparison

Comparing the standalone target-only result with this gamma-1 result:

\[
S=\frac{79.47}{77.33314739037071}\approx1.028
\]

This is an observed throughput ratio of approximately **1.028×**, or about **2.76% higher throughput** for the reported speculative run. This comparison is meaningful only to the extent that the two measurements use comparable timing boundaries and conditions; repeated measurements are needed to determine whether the small difference is repeatable.

---

## 4. Acceptance accounting

Acceptance accounting records the number of speculative blocks that accepted exactly \(k\) draft tokens. This gives more detail than a single overall acceptance ratio.

### Gamma = 1

| Accepted draft tokens in a block | Blocks | Percentage |
|---:|---:|---:|
| 0 | 1,137 | 40.43% |
| 1 | 1,675 | 59.57% |
| **Total** | **2,812** | **100%** |

Mean accepted draft tokens per block:

\[
\bar{k}=\frac{0(1137)+1(1675)}{2812}=0.5957
\]

### Gamma = 2

| Accepted draft tokens in a block | Blocks | Percentage |
|---:|---:|---:|
| 0 | 991 | 42.46% |
| 1 | 530 | 22.71% |
| 2 | 813 | 34.83% |
| **Total** | **2,334** | **100%** |

Mean accepted draft tokens per block:

\[
\bar{k}=\frac{0(991)+1(530)+2(813)}{2334}\approx0.9237
\]

### Gamma = 4

| Accepted draft tokens in a block | Blocks | Percentage |
|---:|---:|---:|
| 0 | 774 | 40.23% |
| 1 | 422 | 21.93% |
| 2 | 299 | 15.54% |
| 3 | 168 | 8.73% |
| 4 | 261 | 13.57% |
| **Total** | **1,924** | **100%** |

Mean accepted draft tokens per block:

\[
\bar{k}=
\frac{0(774)+1(422)+2(299)+3(168)+4(261)}{1924}
\approx1.3347
\]

### Interpretation

- Mean accepted draft tokens per block increases across these measurements: 0.5957 at gamma 1, 0.9237 at gamma 2, and 1.3347 at gamma 4.
- The fraction of blocks accepting every proposal decreases from 59.57% at gamma 1 to 34.83% at gamma 2 and 13.57% at gamma 4.
- Mean accepted draft tokens per block is not the same as total output tokens per block. A speculative block may also emit a correction token or a bonus token.

These distributions should be compared directly with another experiment only when the prompt, sampling configuration, and counting procedure are equivalent.

---

## 5. Phase 2 — Gamma sweep

The Phase 2 experiment reported a target-only baseline of **58.48 tok/s**, with latency of **17.099 ms/token**. The results below preserve that experiment's reported baseline and metrics.

| Gamma | Target tok/s | Spec tok/s | Speedup | Alpha | Draft tokens generated | Accepted tokens | Target verification count | Latency (ms/token) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.48 | 53.44 | 0.914× | 0.5860 | 2,830 | 1,653 | 2,830 | 18.712 |
| 2 | 58.48 | 47.33 | 0.809× | 0.6065 | 4,530 | 2,208 | 2,274 | 21.128 |
| 4 | 58.48 | 29.65 | 0.507× | 0.5897 | 7,900 | 2,491 | 1,998 | 33.730 |
| 6 | 58.48 | 25.31 | 0.433× | 0.6066 | 10,844 | 2,657 | 1,830 | 39.513 |
| 8 | 58.48 | 19.30 | 0.330× | 0.5913 | 14,428 | 2,631 | 1,854 | 51.806 |
| 12 | 58.48 | 17.38 | 0.297× | 0.6075 | 20,510 | 2,711 | 1,779 | 57.523 |
| 16 | 58.48 | 12.27 | 0.210× | 0.5893 | 28,389 | 2,634 | 1,856 | 81.475 |

The experiment produced a speedup-versus-gamma graph and an acceptance-versus-gamma graph.

### Interpretation

1. Gamma 1 is the fastest of the tested speculative configurations in this sweep, at 53.44 tok/s.
2. No gamma in this sweep exceeds its reported target-only baseline of 58.48 tok/s.
3. Alpha remains in a relatively narrow range, approximately 0.586–0.608.
4. Speculative throughput decreases as gamma increases in these measurements.
5. Draft-token proposals increase substantially with gamma, while accepted-token totals do not increase proportionally at the largest gamma values.

For example, gamma 8 reports 14,428 proposed draft tokens and 2,631 accepted tokens; gamma 16 reports 28,389 proposed draft tokens and 2,634 accepted tokens. This suggests that generating more proposals did not yield a corresponding increase in the aggregate number of accepted tokens in this experiment. A complete interpretation also depends on block counts, generated output length, and the exact counting definitions.

**Important baseline note:** the standalone target-only benchmark reports 77.333 tok/s, while this Phase 2 experiment reports 58.48 tok/s as its target-only baseline. These are separate current experiment outputs. The Phase 2 speedups above are calculated against 58.48 tok/s, as reported by that experiment; they should not be recalculated using the standalone baseline unless the runs are confirmed to be directly comparable.

---

## 6. Draft-versus-target decode cost

A separate experiment measured the per-token decoding costs of the two models.

| Metric | Draft model | Target model |
|---|---:|---:|
| Mean latency/token | 5.8217 ms | 14.9830 ms |
| Standard deviation | 2.5261 ms | 2.1405 ms |
| Throughput | 171.77 tok/s | 66.74 tok/s |

The cost coefficient is defined as:

\[
c=\frac{t_{\text{draft}}}{t_{\text{target}}}
\]

Using the reported mean latencies:

\[
c=\frac{5.8217}{14.9830}\approx0.3886
\]

Thus, under this experiment's timing setup, one draft decoding step costs approximately **38.86% of one target decoding step**.

### Interpretation

The coefficient compares the measured costs of individual decoding steps. It does not by itself predict end-to-end speculative speedup. Total speculative performance also depends on verification cost, acceptance behavior, cache operations, sampling, synchronization, and other overheads.

The target latency in this cost experiment is 14.983 ms/token, whereas the Phase 2 baseline reports 17.099 ms/token and the standalone target-only run reports 12.931 ms/token. Keep these values tied to their respective experiments until the timing conventions and baseline comparability are verified.

---

## 7. Speculative block timing

The following experiment measured the time attributed to draft generation, target verification, overhead, and the full speculative block. Timings are in milliseconds per block.

| Gamma | Draft time/block | Target verification time/block | Overhead/block | Total block time | Blocks | Reported tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.972 | 13.595 | 0.833 | 21.400 | 2,800 | 74.60 |
| 2 | 21.269 | 29.662 | 1.687 | 52.617 | 2,318 | 36.65 |
| 4 | 31.351 | 36.434 | 1.717 | 69.503 | 1,977 | 32.53 |
| 6 | 53.424 | 56.414 | 2.507 | 112.345 | 1,911 | 20.82 |
| 8 | 59.334 | 66.160 | 2.019 | 127.514 | 1,849 | 18.96 |
| 12 | 90.361 | 58.859 | 2.407 | 151.627 | 1,815 | 16.24 |
| 16 | 121.114 | 59.370 | 2.519 | 183.003 | 1,797 | 13.59 |

### Arithmetic consistency

The reported total block time is approximately the sum of the three timing components:

\[
T_{\text{block}}=T_d+T_v+T_{\text{overhead}}
\]

For gamma 2, the component sum is 52.618 ms and the reported total is 52.617 ms, a 0.001 ms difference consistent with rounding.

### Observations

- Draft time generally rises with gamma, consistent with proposing more tokens.
- Draft time does not scale exactly linearly with gamma. For example, gamma 1 reports 6.972 ms/block, while gamma 2 reports 21.269 ms/block.
- Target verification time rises through gamma 8, then is lower at gamma 12 and gamma 16.
- Total block time increases with gamma in these measurements.
- Reported throughput declines as gamma increases after gamma 1.

Non-linear timing does not automatically indicate a bug: execution and synchronization costs can depend on workload size. However, the non-linear component measurements should be checked against the timing instrumentation and repeated before being used to make detailed claims about bottlenecks.

### Timing caveat

The reciprocal of block time gives blocks per second, not tokens per second. Since a block can produce a variable number of output tokens, throughput must be calculated using the actual number of output tokens generated over the full timed interval.

Before treating the component timings as a definitive cost breakdown, verify that:
- Each timer includes the intended MLX evaluation/synchronization.
- Timing boundaries are consistent across gamma values.
- The component timers account for the same work as the total block timer, without unintended gaps or overlaps.
- The throughput calculation uses the actual timed output-token count.
- “Blocks,” “target verification count,” and “draft tokens generated” have clearly defined and consistent meanings.

---

## 8. Findings from the current measurements

The current results support the following observations:

1. The standalone target-only benchmark reports 77.333 tok/s.
2. The standalone gamma-1 speculative benchmark reports 79.47 tok/s, an observed throughput ratio of approximately 1.028× relative to that standalone baseline.
3. The Phase 2 sweep uses its own reported target-only baseline of 58.48 tok/s; none of its tested gamma values exceeds that baseline.
4. Acceptance accounting for gamma 1, 2, and 4 gives mean accepted draft-token counts per block of 0.5957, 0.9237, and 1.3347, respectively.
5. In the Phase 2 sweep, alpha stays around 0.59–0.61 while throughput declines with larger gamma.
6. The separate draft-versus-target cost experiment reports \(c=0.3886\).
7. The block-timing experiment shows rising total block time with gamma, while some individual timing components vary non-linearly and require validation.

These observations apply to the reported experiments. Results from different sections should not be combined into one performance claim unless their configurations and timing conventions are known to be comparable.

---

## 9. Next step — consolidate the experimental harness

The next planned stage is to consolidate the existing experiments into a reproducible harness.

### Baseline and run metadata
- [ ] Record per-run target-only and speculative timing, not just averages.
- [ ] Record model identifiers/revisions, Python executable, and package versions.
- [ ] Keep prompts, token counts, sampling parameters, warmups, and measurement runs fixed for direct comparisons.
- [ ] Clearly identify which target-only baseline is used for each speedup calculation.

### Acceptance accounting
- [ ] Record counts for every possible accepted-token count from 0 through gamma.
- [ ] Store per-run alpha and aggregate alpha separately.
- [ ] Track proposed draft tokens, accepted draft tokens, correction/bonus tokens, and block counts as distinct quantities.

### Timing validation
- [ ] Review timer boundaries for draft generation, target verification, overhead, and total block time.
- [ ] Confirm MLX computations are evaluated before the relevant timers stop.
- [ ] Validate token accounting used for throughput.
- [ ] Repeat timing measurements and retain variability statistics.
- [ ] Ensure component timings are interpreted consistently.

### Harness outputs
For every gamma, the final harness should save:
- Target-only baseline and speculative throughput.
- Latency and total elapsed time.
- Speedup.
- Alpha and acceptance-accounting distributions.
- Draft-token, accepted-token, and verification counts.
- Draft, verification, overhead, and total block timings.
- Speedup-versus-gamma and acceptance-versus-gamma plots.
- Machine-readable results for later analysis.

---

## Current status

**Completed:** target-only benchmarking, gamma-1 speculative benchmarking, acceptance accounting for gamma 1/2/4, the Phase 2 gamma sweep, draft-versus-target cost measurement, and speculative block timing.

**Still to validate:** comparability of the separate reported target-only baselines and the timing boundaries used for the block-level breakdown.

**Next milestone:** validate measurement definitions and consolidate the existing experiments into a reproducible experimental harness before changing the model pair.
