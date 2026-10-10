# Speculative Decoding on Apple Silicon — Experimental Log

This repository documents the implementation and measurements completed so far for speculative decoding with MLX-LM on an Apple M3 MacBook Air. It is an experiment log for this specific project: the model pair, benchmark configurations, measured results, observations, and unresolved questions.

The next planned step is to consolidate the existing experiments into a proper experimental harness. Before doing that, this document records the evidence collected so far so that future changes can be compared against it.

---

## 1. Project setup

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
| Target | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Draft | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |

### Main benchmark configuration

The reported speculative benchmark used:

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

The target-only and speculative benchmarks were intended to use the same prompt, generated-token count, sampling settings, hardware, and run counts. The results below are kept in separate experiment sections because the measured target-only baseline changed across runs and scripts.

---

## 2. Work completed so far

The work has been carried out in the following order:

1. Measured target-only generation throughput and latency.
2. Measured speculative decoding throughput and latency, including a gamma-1 run.
3. Recorded acceptance accounting for gamma values 1, 2, and 4.
4. Ran a gamma sweep over `1, 2, 4, 6, 8, 12, 16`, recording acceptance, token counts, throughput, latency, and speedup.
5. Measured the relative cost of draft decoding versus target decoding, obtaining the cost coefficient `c`.
6. Instrumented speculative decoding to measure draft time, target verification time, overhead, and total block time.

The next planned step is to review these measurements and then consolidate the existing experiment code into a reproducible experimental harness.

---

## 3. Target-only baseline measurements

Several target-only throughput values have been observed in different runs. They should **not** be treated as one consistent baseline or averaged together: they came from different benchmark runs, and the reason for the variation has not yet been established.

| Experiment / observation | Target throughput | Latency per token | Notes |
|---|---:|---:|---|
| Earlier historical measurement | 75.60 tok/s | 13.23 ms/token | Earlier result; original exact run conditions have not been recovered |
| `target_only.py` rerun | 77.333 tok/s | 12.931 ms/token | Close to the historical result |
| Phase 2 gamma-sweep baseline | 58.48 tok/s | 17.099 ms/token | Baseline used for the Phase 2 speedup table |
| Earlier `experimental_harness.py` baseline | 41.01 tok/s | About 24.38 ms/token | Separate result from the earlier harness run |
| Draft-versus-target cost experiment | 66.74 tok/s | 14.983 ms/token | Target decode measurement used to calculate `c` |

### Reflection

The target-only results vary considerably, from about 41 tok/s to 77 tok/s. The 77.33 tok/s result is close to the historical 75.60 tok/s result, but the later Phase 2 baseline was 58.48 tok/s. Comparing results from different runs without understanding this variation could distort the calculated speedup.

An investigation compared system Python and the project virtual environment, including MLX versions 0.31.2 and 0.32.1, and tried different synchronization patterns. Those tests did not establish the cause of the throughput variation.

**Current status:** the variation is documented, but its cause remains unresolved. Future comparisons should use the target-only baseline measured under the same controlled run conditions as the corresponding speculative experiment.

---

## 4. Initial speculative measurement — gamma = 1

One speculative benchmark reported the following:

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
| Realized alpha (mean of per-run alphas) | 0.5957 |
| Accepted draft tokens | 1,670 |
| Proposed draft tokens | 2,815 |

The user confirmed that realized alpha is calculated as the arithmetic mean of the acceptance ratio from each measurement run:

\[
\bar{\alpha} = \frac{1}{R}\sum_{i=1}^{R}\alpha_i
\]

where \(R=30\). This is not necessarily equal to the aggregate ratio of all accepted tokens divided by all proposed tokens, because the aggregate ratio weights runs according to their proposal counts.

For this run, the aggregate ratio from the totals is:

\[
\alpha_{\text{aggregate}}=\frac{1670}{2815}\approx 0.5932
\]

This does not contradict the reported mean per-run alpha of 0.5957; they are different aggregation methods.

The target-only `target_only.py` rerun reported 77.333 tok/s. Using these two reported throughputs gives an **observed cross-run comparison** of:

\[
\frac{79.47}{77.333}\approx 1.028
\]

That is approximately 2.76% higher throughput for the speculative run. Both benchmarks were reported to use the same experimental conditions, but this small observed difference should not be treated as conclusive evidence of a repeatable speedup until run-to-run variation is characterized.

This result must also remain separate from the later Phase 2 sweep, which used a target-only baseline of 58.48 tok/s.

---

## 5. Acceptance accounting — gamma = 1, 2, and 4

Acceptance accounting records how many draft tokens were accepted in each speculative block, rather than only reporting one overall alpha value.

### Gamma = 1

| Accepted draft tokens in a block | Number of blocks | Percentage |
|---:|---:|---:|
| 0 | 1,137 | 40.43% |
| 1 | 1,675 | 59.57% |
| **Total** | **2,812** | **100%** |

Mean accepted draft tokens per block:

\[
\frac{0(1137)+1(1675)}{2812}=0.5957
\]

### Gamma = 2

| Accepted draft tokens in a block | Number of blocks | Percentage |
|---:|---:|---:|
| 0 | 991 | 42.46% |
| 1 | 530 | 22.71% |
| 2 | 813 | 34.83% |
| **Total** | **2,334** | **100%** |

Mean accepted draft tokens per block:

\[
\frac{0(991)+1(530)+2(813)}{2334}\approx0.9237
\]

### Gamma = 4

| Accepted draft tokens in a block | Number of blocks | Percentage |
|---:|---:|---:|
| 0 | 774 | 40.23% |
| 1 | 422 | 21.93% |
| 2 | 299 | 15.54% |
| 3 | 168 | 8.73% |
| 4 | 261 | 13.57% |
| **Total** | **1,924** | **100%** |

Mean accepted draft tokens per block:

\[
\frac{0(774)+1(422)+2(299)+3(168)+4(261)}{1924}\approx1.3347
\]

### What these measurements show

- The mean number of accepted draft tokens per block rises across the measured configurations: 0.5957 at gamma 1, 0.9237 at gamma 2, and 1.3347 at gamma 4.
- The probability of accepting every proposal falls as gamma increases in these measurements: 59.57%, 34.83%, and 13.57%, respectively.
- Mean accepted draft tokens per block is not the same as total output tokens per block. Speculative decoding may also emit a correction token or a bonus token.

These counts were collected for the acceptance-accounting experiments. They should not automatically be treated as the exact acceptance distributions for the later Phase 2 sweep unless the prompt, settings, and measurement population match.

---

## 6. Phase 2 — Gamma sweep

The Phase 2 sweep used the following target-only baseline:

- **Target throughput:** 58.48 tok/s
- **Target latency:** 17.099 ms/token

The results below are the reported Phase 2 measurements.

| Gamma | Target tok/s | Spec tok/s | Speedup | Alpha | Draft tokens generated | Accepted tokens | Target verification count | Latency (ms/token) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.48 | 53.44 | 0.914× | 0.5860 | 2,830 | 1,653 | 2,830 | 18.712 |
| 2 | 58.48 | 47.33 | 0.809× | 0.6065 | 4,530 | 2,208 | 2,274 | 21.128 |
| 4 | 58.48 | 29.65 | 0.507× | 0.5897 | 7,900 | 2,491 | 1,998 | 33.730 |
| 6 | 58.48 | 25.31 | 0.433× | 0.6066 | 10,844 | 2,657 | 1,830 | 39.513 |
| 8 | 58.48 | 19.30 | 0.330× | 0.5913 | 14,428 | 2,631 | 1,854 | 51.806 |
| 12 | 58.48 | 17.38 | 0.297× | 0.6075 | 20,510 | 2,711 | 1,779 | 57.523 |
| 16 | 58.48 | 12.27 | 0.210× | 0.5893 | 28,389 | 2,634 | 1,856 | 81.475 |

The experiment produced speedup-versus-gamma and acceptance-versus-gamma graphs.

### Main observations

1. **Gamma = 1 was the fastest tested speculative configuration:** 53.44 tok/s.
2. **None of the tested gamma values exceeded the Phase 2 target-only baseline** of 58.48 tok/s.
3. Alpha stayed in a fairly narrow range, approximately 0.586–0.608.
4. Speculative throughput declined as gamma increased in this sweep.
5. The number of proposed draft tokens increased substantially with gamma, while the accepted-token total did not increase proportionally at the larger gamma values.

For example, gamma 8 recorded 14,428 draft tokens and 2,631 accepted tokens, while gamma 16 recorded 28,389 draft tokens and 2,634 accepted tokens. These are aggregate counts from the reported experiment; interpreting them fully requires the corresponding block counts and exact counting definitions.

### Interpretation

The measured acceptance ratio alone does not explain throughput. The time spent generating draft tokens, verifying them, and handling the speculative block also matters.

These results show that, for this model pair and this implementation under the reported Phase 2 conditions, larger gamma values were not beneficial for end-to-end throughput. They do not establish that the same behavior will occur for every model pair or implementation.

---

## 7. Phase 3 — Relative draft/target decoding cost

A separate experiment measured the decoding cost of the draft and target models.

| Metric | Draft model | Target model |
|---|---:|---:|
| Mean latency/token | 5.8217 ms | 14.9830 ms |
| Standard deviation | 2.5261 ms | 2.1405 ms |
| Throughput | 171.77 tok/s | 66.74 tok/s |

The measured cost coefficient was defined as:

\[
c=\frac{t_{\text{draft}}}{t_{\text{target}}}
\]

Using the reported mean latencies:

\[
c=\frac{5.8217}{14.9830}\approx0.3886
\]

Therefore, the measured draft decode step costs about **38.86% of the measured target decode step** under this experiment's timing setup.

### Interpretation and caveat

This coefficient is useful for comparing the individual decode-step costs. It does not by itself predict end-to-end speculative speedup. Speculative decoding also incurs target verification and implementation overhead, and verification cost can change with gamma.

The cost-coefficient experiment's target latency (14.983 ms/token) differs from the Phase 2 baseline latency (17.099 ms/token) and from the 12.931 ms/token result from `target_only.py`. The measurements should remain separate until the cause of baseline variation is understood and the timing methods are confirmed to be comparable.

The draft latency standard deviation is also relatively large compared with its mean, so the coefficient should be treated as an estimate from this run rather than a perfectly stable constant.

---

## 8. Phase 3 — Speculative block timing

A separate timing experiment measured the time attributed to draft generation, target verification, overhead, and the complete speculative block.

The reported table is reproduced below. All timing columns are in milliseconds per block.

| Gamma | Draft time/block | Target verification time/block | Overhead/block | Total block time | Blocks | Reported tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.972 | 13.595 | 0.833 | 21.400 | 2,800 | 74.60 |
| 2 | 21.269 | 29.662 | 1.687 | 52.617 | 2,318 | 36.65 |
| 4 | 31.351 | 36.434 | 1.717 | 69.503 | 1,977 | 32.53 |
| 6 | 53.424 | 56.414 | 2.507 | 112.345 | 1,911 | 20.82 |
| 8 | 59.334 | 66.160 | 2.019 | 127.514 | 1,849 | 18.96 |
| 12 | 90.361 | 58.859 | 2.407 | 151.627 | 1,815 | 16.24 |
| 16 | 121.114 | 59.370 | 2.519 | 183.003 | 1,797 | 13.59 |

### Arithmetic check

The reported total block time is consistent with the sum of the three reported components, within rounding:

\[
T_{\text{block}}=T_d+T_v+T_{\text{overhead}}
\]

For gamma 2:

\[
21.269+29.662+1.687=52.618\text{ ms}
\]

The table reports 52.617 ms, a 0.001 ms difference attributable to rounding.

### Observations requiring investigation

- Draft time generally increases with gamma, as more draft tokens are proposed.
- Draft time does not scale perfectly linearly with gamma. For example, gamma 1 reports 6.972 ms/block, while gamma 2 reports 21.269 ms/block.
- Target verification time rises from gamma 1 through gamma 8, then is lower for gamma 12 and gamma 16.
- Total block time increases with gamma in these measurements.
- Reported throughput decreases with gamma after gamma 1.

These patterns do not automatically prove a timing bug. GPU execution, synchronization, workload size, and timer boundaries can affect scaling. However, the results justify checking the timing code and repeating measurements before drawing detailed conclusions about the cost of individual phases.

### Timing interpretation caveat

The reciprocal of block time gives blocks per second, not tokens per second. A block can yield different numbers of output tokens depending on acceptance and correction/bonus behavior. Therefore, the reported token throughput must be calculated from the actual output-token count and the full timed generation interval.

Before using the timing table to explain throughput, verify that:

- Each component timer includes the intended MLX evaluation/synchronization.
- Timer boundaries are consistent across gamma values.
- The component timings cover the same work as the total block timer without unintended gaps or overlaps.
- The token count used for throughput matches the output tokens generated in the timed interval.
- The meaning of “target verification count” and “blocks” is consistent across experiments.

The timing table is retained as a measurement, but its component definitions need to be validated before treating it as a final cost breakdown.

---

## 9. Combined findings so far

The measurements collected to date support these conclusions:

1. A target-only benchmark has produced results from approximately 41 to 77 tok/s across separate experiments. The cause of this variation has not been established.
2. One gamma-1 speculative run reported 79.47 tok/s against a target-only run reporting 77.33 tok/s, an observed cross-run ratio of about 1.028×. This small difference is not yet evidence of a repeatable speedup.
3. The separate Phase 2 sweep used a 58.48 tok/s target-only baseline. None of its tested gamma values exceeded that baseline.
4. Acceptance accounting for gamma 1, 2, and 4 is internally consistent with the reported block counts and mean accepted-token counts.
5. In Phase 2, alpha remained around 0.59–0.61 while throughput declined as gamma increased.
6. The measured cost coefficient was \(c=0.3886\), based on the separate draft-versus-target decode-cost experiment.
7. The timing breakdown shows increasing total block time with gamma, but some component timings have non-linear behavior that should be checked against the instrumentation.

These are results from different runs and experimental stages. They should not be combined into one performance claim unless their configurations and timing definitions are confirmed to be comparable.

---

## 10. Work remaining before the experimental harness

The next step is to consolidate the existing experiment code into a proper experimental harness. Before doing so, the following checks will make the resulting measurements easier to trust and reproduce.

### A. Standardize baseline measurement

- [ ] Use one documented target-only timing convention.
- [ ] Save individual run times as well as mean and standard deviation.
- [ ] Record the exact model revision and Python/package versions.
- [ ] Ensure target-only and speculative comparisons use comparable prompts, token counts, sampling settings, warmups, and measurement runs.
- [ ] Investigate the observed target-only throughput variation.

### B. Standardize acceptance accounting

- [ ] Record the number of blocks accepting each possible number of draft tokens.
- [ ] Clearly distinguish the mean of per-run alpha values from aggregate alpha.
- [ ] Record proposed tokens, accepted tokens, correction/bonus output tokens, and block counts separately.

### C. Validate the timing instrumentation

- [ ] Inspect the timer boundaries for draft generation, target verification, overhead, and total block time.
- [ ] Confirm the appropriate MLX evaluation/synchronization occurs before timers stop.
- [ ] Repeat the gamma timing experiment to check variability.
- [ ] Verify output-token accounting used for throughput.
- [ ] Check whether the timing components cover the full block without double counting.

### D. Consolidate into the experimental harness

The harness should produce, for each gamma:

- Target-only baseline and speculative throughput.
- Latency and total timed duration.
- Speedup.
- Alpha and acceptance accounting.
- Draft-token, accepted-token, and verification counts.
- Draft, verification, overhead, and total block timings.
- The speedup-versus-gamma and acceptance-versus-gamma graphs.
- Machine-readable results for later analysis.

The harness should preserve per-run data so that averages, variability, and derived metrics can be checked after the experiment.

---

## 11. Current project status

**Completed:** target-only benchmarking, speculative benchmarking, acceptance accounting for gamma 1/2/4, the Phase 2 gamma sweep, cost-coefficient measurement, and speculative block timing instrumentation.

**Not yet resolved:** variation in target-only throughput across separate runs and the detailed interpretation of some component timings.

**Next milestone:** validate the measurement conventions and timing instrumentation, then consolidate the existing experiments into a reproducible experimental harness.

The goal of the next stage is not to change models immediately. It is to make the current measurements reproducible and internally interpretable so that future model or implementation comparisons are based on reliable evidence.
