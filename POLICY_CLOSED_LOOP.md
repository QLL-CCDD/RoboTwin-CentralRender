# Real RoboTwin K2/G1 Policy closed loop — 2026-11-12

The requested real closed loop is complete: both modes finished two episodes,
both succeeded, and their complete 234-action sequences are identical. The
first pair observes higher aligned formal throughput with centralized rendering,
but **does not establish a stable rendering-attributed or end-to-end speedup**.
First-inference/cache sensitivity is material. Peak device VRAM is lower.

## All completed runs, in chronological order

The selected pair is native-k2g1-r2 followed by central-k2g1-r3. The later
native-k2g1-r3 was launched to investigate the first-inference discrepancy and
completed before the stop instruction was processed. It is retained as
**unpaired sensitivity evidence**. No further centralized repeat was launched.

| Metric | Native selected pair | Central selected pair | Later native sensitivity |
| --- | ---: | ---: | ---: |
| Episodes / successes | 2 / 2 | 2 / 2 | 2 / 2 |
| Steps by seed 1000000 / 1000001 | 110 / 124 | 110 / 124 | 110 / 124 |
| Total evaluated actions | 234 | 234 | 234 |
| Real model forward calls | 15 | 15 | 15 |
| Four-camera observations / RGBs | 17 / 68 | 17 / 68 | 17 / 68 |
| Formal wall (s) | 6.831210 | 5.658804 | 6.031630 |
| Formal steps/s | 34.254549 | 41.351497 | 38.795482 |
| Total wall including load/preparation/teardown (s) | 29.538470 | 29.631902 | 28.613666 |
| Policy loading (s) | 11.139696 | 11.018539 | 11.308012 |
| First model inference (s) | 1.970766 | 0.546450 | 0.628985 |
| Mean subsequent 14 model inferences (ms) | 59.051 | 63.966 | 63.340 |
| Peak device VRAM (MiB) | 16708 | 13956 | 16708 |

Every completed episode terminated by success, at 110 and 124 steps respectively;
the step limit is 400. The sensitivity run also has identical action sequences.
All runs have K=2 concurrent simulator processes and G=1 actual Policy process.
The raw SDK records also report 234 evaluated steps in every run. These are
qpos action steps, distinct from the 15 model-forward calls and from internal
250 Hz physics substeps, whose total count was not instrumented.

For the selected first pair:

- Observed formal throughput change: **+20.718267%**,
  calculated as central steps/s divided by native steps/s, minus one.
- Observed formal wall reduction: **17.162495%**,
  calculated as one minus central formal time divided by native formal time.
- Total wall reduction: **-0.316303%**; central wall
  is 0.093431 s longer (about 0.3%).
  That small difference is not interpreted as a significant regression.
- Peak device VRAM decreases by **2,752 MiB (16.471%)**.

The first native inference is 1.424316 s slower
than central, while the formal wall differs by only
1.172406 s. This is a material startup/cache confound.
The later native first inference is 0.629 s and its observed rate is
38.795 steps/s. These are measured observations, not a fabricated adjusted
rendering rate: overlapping inference, physics and queue spans are not
subtracted from the wall time. Only two episodes and approximately six-second
formal windows were measured; stable throughput and success-rate improvements
are unproven.

## Timing boundaries

Formal start is the common barrier release after both simulators have their
first real four-camera observation and before either sends its first actual
Policy step. Formal end is the last completed episode-end callback. First model
inference remains inside this window in both modes. There is no unscored model
warmup, replay or fixed-action substitute.

Total wall starts before launching either the native CLI or centralized service,
and ends after command/service teardown. It includes fresh model load, render
startup, physical setup and first-frame preparation. The original SDK evaluation
window is retained separately; it includes simulator initialization and is not
the aligned formal window.
For completeness, the SDK's result elapsed/rate fields are 18.197830 s /
12.858676 steps/s (native selected), 12.055796 s / 19.409751 steps/s (central),
and 17.110052 s / 13.676171 steps/s (native sensitivity). Those fields include
worker preparation and cleanup, and use a different boundary from both the
aligned formal window and the full command wall time. Service startup occurs
before the SDK process in central mode, so these raw SDK rates are not used
as the primary like-for-like rendering comparison.

All timestamps are Asia/Shanghai (UTC+08:00), 2026-11-12.

| Run | Formal start | Formal end | Original SDK evaluation wall (s) |
| --- | --- | --- | ---: |
| native-k2g1-r2 | 2026-11-12T12:40:08.159+08:00 | 2026-11-12T12:40:14.990+08:00 | 16.829425 |
| central-k2g1-r3 | 2026-11-12T12:47:47.794+08:00 | 2026-11-12T12:47:53.453+08:00 | 11.019895 |
| native-k2g1-r3 | 2026-11-12T12:51:09.873+08:00 | 2026-11-12T12:51:15.905+08:00 | 15.626380 |

## Fixed workload and actual environment

- adjust_bottle / demo_clean / aloha-agilex; qpos, fast control=1, shared native
  engine=1 where applicable, verified frozen seed cache, unchanged native cache
  clearing frequency, MPS disabled.
- Four D435 cameras, 320x240, RT32 samples/pixel, path depth 8, OIDN. The Policy
  consumes head/left/right images; front is still rendered. Action horizon and
  observation stride are 16. Terminal observations are retained. Video, point
  clouds and depth are disabled.
  The existing research subset runner uses instruction seed limit 2 in all arms.
- Host autodl-pro-79387867b0ad, RTX4090 as reported by the driver (49,140 MiB),
  UUID GPU-9c9d4cd7-338d-dd8e-ff4f-85959802fc6f, driver 580.105.08.
  No other GPU job overlaps the measured runs.
- Controller Python 3.11.15; existing Policy and simulator Python 3.10.20;
  SDK 0.8.0b31, Torch 2.6.0+cu124, Transformers 4.57.0, FlashAttention
  2.7.4.post1. Python 3.10 Policy use was expressly permitted for this experiment.
  The frozen SDK is imported without source changes; its declared Python >=3.11
  support is not changed by this experimental execution.
- StarVLA 7e4a47360093d79dc42b883885c04fbe790c5b60.
  Model StarVLA/Qwen3-VL-OFT-Robotwin2 at
  e9cdfc38a0db968bf3a865bcb542d3d455952226, 9,785,132,794 bytes.
  Weight SHA256:
  8ccee8ddb33fa5df83a9a5e2b7d4f737db93f19a051b23f738af8a484fe2d7e5.
  It was verified on D and by the cloud downloader. Exact D config, statistics
  and Qwen metadata were copied. The native recipe omits state input, uses
  bfloat16 and retains the original inference kwargs.
- Official RoboTwin c3ddfa8b97d5519efa828b075999bd0006778e5e plus the official
  builder's deterministic annotation patch reproduces runtime commit
  0fe7d15a86ae19d9411a52520683039cfc81d9ff exactly. Existing aloha/bottle assets
  are reused. No CUDA or physics stack was rebuilt.

## Actual process and backend verification

| Run | Simulator PIDs | Actual Policy PID | Render service PID | GPU roles |
| --- | --- | --- | --- | --- |
| Native selected | 21502, 21503 | 21437 | none | both simulators have graphics; one Policy compute process |
| Central selected | 23581, 23582 | 23517 | 23322 | only service has graphics; physics workers absent from GPU |
| Native sensitivity | 24003, 24004 | 23935 | none | both simulators have graphics; one Policy compute process |

The service reports two distinct visual scenes without physics and a shared
native context. Resource logs include graphics and compute processes; duplicate
NVIDIA rows are deduplicated by PID. Runtime readiness asserts no renderer or
RenderSystem in central physics workers, and both are present in native mode.
Each run contains 15 real model forwards, real physical actions and updated
observations. No silent native fallback is present.

## Repairs and preserved attempts

native-k2g1-r1 loaded the model but completed no episodes: isolated -I SDK
workers ignored PYTHONPATH. The private controller now has an SDK .pth entry.

central-k2g1-r1 and r2 completed no episodes. The crash stack identifies the
official eval entry's independent Sapien_TEST render smoke scene before task
setup. CPU workers now delegate that GPU preflight to the already-ready service.
The task's physics/control and four-camera quality are preserved. The concurrent
physics-URDF cache write was also made atomic; the observed crash specifically
came from the smoke renderer.

Failed attempts are kept separately and are not folded into successful episode
counts or timings. The later completed native run is reported as sensitivity
evidence, not as a failure or hidden exclusion. Its initial selection note and
corrected sensitivity annotation both remain in the raw archive.

## Re-run on the prepared cloud host

Prepared root is /root/autodl-tmp/roboscope-central-render-20261111.
Choose new run names; the script refuses existing directories or occupied GPUs.
These are the actual fixed K2/G1 entry commands, not a request to rerun them.

~~~bash
base=/root/autodl-tmp/roboscope-central-render-20261111
bash "$base/run_policy_arm.sh" native new-native-name
bash "$base/run_policy_arm.sh" central new-central-name
"$base/policy-env/bin/python" "$base/analyze_policy_pair.py" \
  "$base/closed-loop/new-native-name" "$base/closed-loop/new-central-name" \
  --output "$base/new-comparison.json"
~~~

The wrapper uses the stock SDK subset runner with subset-task-count=1,
subset-episodes=2, episode-shard-size=1, concurrency=2 and replicas=1.
Both modes use the same manifest, weights and Policy config. policy_run.py
owns the first-observation barrier; native_entry.py switches the RGB backend;
central_launch.py owns the service lifecycle. SDK source and shared installed
simulator environments were not patched.

[closed-loop-result.json](closed-loop-result.json) contains exact percentages,
PIDs, timestamps and all 15 inference durations per run.
[closed-loop-provenance.json](closed-loop-provenance.json) records source hashes.
Raw SDK, Policy, action, resource and failure logs are in
evidence/policy-closed-loop-evidence.tar.gz, extracted at evidence/policy-cloud/.
The latest code is central-render-closed-loop-bundle.tar.gz. Credentials and
model weights are excluded.

Stage 1's +25.14% nominal visual replay and stage 2's +19.92% synthetic dynamic
IPC replay are separate workloads. Stage 3 establishes sampled physical/image
equivalence. They are not extra Policy measurements and are not pooled here.
