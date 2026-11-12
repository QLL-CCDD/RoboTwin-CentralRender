# RoboTwin centralized rendering experiment

**The real K2/G1 Policy closed loop is now complete.** Both modes finished two
episodes successfully with the same 234 actions. The first pair observed
34.255 -> 41.351 steps/s (+20.718%), but the native first inference was slower
(1.971 vs 0.546 s), and a later, already-completed native sensitivity run reached
38.795 steps/s. These short windows do not establish stable rendering-attributed
or total-wall acceleration. Total wall was 29.538 vs 29.632 s, essentially equal;
peak VRAM was 16,708 vs 13,956 MiB. All completed runs, timing boundaries,
first/subsequent inference times and reproducible commands are in
[POLICY_CLOSED_LOOP.md](POLICY_CLOSED_LOOP.md) and
[closed-loop-result.json](closed-loop-result.json). Latest code is packaged in
`central-render-closed-loop-bundle.tar.gz`; raw logs are preserved separately in
`evidence/policy-closed-loop-evidence.tar.gz`.

Stage 3 established real CPU RoboTwin physics with remote RGB. On the idle cloud
GPU, the two modes produced identical physical snapshots, camera parameters and
64 paired RGB images across initial state, six actions per worker and reset.
Its historical `stage3-result.json` correctly records that Policy had not run
at that stage. Stage 1 and 2 throughput numbers below describe different visual
workloads and do not measure rollout speedup.

## Stage 1: nominal visual replay

Stage 1 measured result (2026-11-11): **+25.14% image throughput**, with equal
800-image workloads and 16 cross-arm image samples exactly equal. This is one
short fixed nominal visual replay, not a rollout or IPC result.

| N=2, four 320x240 cameras, RT32/depth8/OIDN | Independent processes | One shared native context |
| --- | ---: | ---: |
| Timed observations / RGB images | 200 / 800 | 200 / 800 |
| Steady elapsed seconds | 10.6573 | 8.5165 |
| Images/s | 75.0661 | 93.9357 |
| Observations/s | 18.7665 | 23.4839 |
| Synthetic request completion mean, ms | 106.55 | 82.03 |
| Completion p95, ms | 110.27–110.63 per worker | 88.39 |
| Steady device VRAM mean/sample peak, MiB | 6480 / 6480 | 3727 / 3727 |

Measured host: `autodl-pro-79387867b0ad`, Xeon Gold 6459C, RTX4090 reporting
49140 MiB, GPU `GPU-9c9d4cd7-338d-dd8e-ff4f-85959802fc6f`, driver 580.105.08.
Python 3.10.20, NumPy 1.26.4 and SAPIEN 3.0.0b1; existing Torch 2.6.0 is not a
Policy benchmark. Small missing rendering dependencies were installed into the
isolated `site/` directory. No planning stack was installed for stage 1.

The replay uses the complete aloha-agilex URDF and all its referenced meshes and
textures, native bottle models 13/16, and unchanged pinned upstream camera,
table and object constructors. Four cameras include front_camera. The robot is
at a nominal home state with one initialization physics step; warmup and timing
contain no physics. Wrist views in this particular home state mostly see the
wall. This is not `setup_demo`'s settled/expert-valid state or a diverse task mix.

Asset ZIP entries were fetched from pinned official RoboTwin archives using
bounded HTTP ranges: 90,827,619 compressed member bytes in 25.42 s, without
downloading unrelated models. Source URLs, ZIP member identities and file
SHA256 values are in `evidence/cloud/replay-assets-manifest.json`.

Results and images: `evidence/cloud/results/n2-r1/`. Both scene-state and source
checks pass. The 16 checked warm/last RGB images have MAE 0 and 100% exact pixels;
both scene contact sheets were produced. Native engine/context identity asserts
pass; scene render systems remain distinct. The same native context/device and
resource manager are shared, while cameras retain their own RT render targets.
This supports consolidation, but does not isolate Vulkan context switching as
the cause or predict whole-rollout/48-worker speedup. IPC is not included.

The GPU returned to 0 MiB / 0% with no compute processes after the run. No other
host's performance is pooled with this result. No larger N or MPS grid was run.

Actual cloud entry point, already staged at
`/root/autodl-tmp/roboscope-central-render-20261111`:

```bash
bash /root/autodl-tmp/roboscope-central-render-20261111/run_cloud_pair.sh n2-r1 2 100
```

## Stage 2: dynamic state workers and IPC

The measured order was **central first, independent second**. Both arms rendered
the same 200 dynamic observations / 800 RGB images, with five untimed warmups per
worker and a separate post-timing reset check. All four cameras are retained.

| N=2 dynamic replay, RT32/depth8/OIDN | Independent GPU workers | One GPU service + two CPU workers |
| --- | ---: | ---: |
| Steady elapsed seconds | 12.2172 | 10.1880 |
| Images/s | 65.4815 | 78.5238 |
| Client request mean/p95, ms | 120.63 / 123.13 | 96.10 / 99.20 |
| Steady VRAM mean/sample peak, MiB | 6484 / 6484 | 3729 / 3729 |

Throughput improves **19.92%** in this one short pair. The trajectory is synthetic
CPU URDF forward kinematics, with wrist cameras aimed toward the bottle. It is
not a Policy rollout. Each visual scene contains 56 bodies and 60 render shapes,
cloned from the complete native assets. No physics system is attached to these
service scenes. Workers transmit body/link poses, robot qpos and camera poses.

State RPCs use local Unix connections; four RGB arrays per scene use shared
memory. Replies carry worker, scene, episode, generation and frame IDs and a
state digest. Reset rebuilds the visual scene/cameras and restarts the frame
sequence under a new generation. Shutdown closes connections and unlinks shared
memory. The measured stage 2 reset restored a fixed asset template; stage 3
adds actual native model specifications for asset-changing resets.

All 200 timed input/identity records match across modes. All 32 sampled dynamic
and reset RGB images are exactly equal, with MAE 0. Every camera changes across
the sampled motion and reset. Two central CPU workers did not import SAPIEN and
do not appear among GPU processes; the only unique measured GPU PID is the
service. The driver repeats graphics/CUDA PID rows, so ownership is checked by
unique PID. No rendering/performance rerun was needed for this parser correction.

Central service-inclusive means are 48.05/51.12 ms per worker, and measured queue
after service recv is about 1.08 ms. The remaining client time, 44.03–46.86 ms,
includes waiting before service recv as well as serialization, transport and
shared-memory copies. It is not a measured pure communication cost.

```bash
bash /root/autodl-tmp/roboscope-central-render-20261111/run_cloud_ipc.sh ipc-n2-next 100
```

Results: `evidence/cloud/results/ipc-n2-r1/`. `central_render_service.py` exposes
the GPU process entry point and CPU-only `RenderClient`; `benchmark_ipc.py` shows
the complete start/attach/reset/render/close/shutdown lifecycle. The camera and
state facade in `robotwin_worker_adapter.py` preserves native `get_obs` packing
for a task already constructed as physics-only. Stage 3 below validates this
path with real physical actions; Policy integration remains unmeasured.

Cloud readiness at the end of stage 2: `/root/starVLA` at `7e4a4736` and its Python environment
contain StarVLA, Transformers, Accelerate and FlashAttention. The usual model/HF
cache directories checked here contain no verified checkpoint. SAPIEN is ready
in the isolated renderer site directory; native gymnasium, mplib, TOPPRA,
cuRobo and Open3D were absent from the base Policy environment. Stage 3 added
the required CPU packages in the isolated site directory, without cuRobo or
any model download.

This isolated experiment is based on RoboScope Kit main
`329ef983ee81c4f8889e81393c8b7a42dc8c595e` (2026-11-10), checked out in `kit/`
on `codex/robotwin-central-render-20261111`. The checkout is unchanged; no commit,
PR, release or installed simulator patch was made. The older local PR98 checkout
and its documentation changes were preserved.

## Stage 3: real RoboTwin physics and remote observations

Supported scope is **adjust_bottle / demo_clean / aloha-agilex / qpos / RGB**.
All four cameras remain 320x240 D435, RT32, path depth 8, OIDN. No depth,
segmentation, point cloud, video, other tasks, or expert cuRobo planning is
supported by this experimental backend.

Both modes run original task initialization, including the native settling
and stability check. The native Mplib TOPP, collision meshes, link masses,
joint drives, gripper interpolation, physics substeps and success checks remain
in use. Both modes use the same existing Kit qpos control patches. CPU workers
load a copy of the URDF with visual elements removed, create PhysxCpuSystem
without RenderSystem, use metadata cameras with the original random draws,
and suppress visual builders and local GPU-cache clearing. They import SAPIEN
and Torch for simulation, but do not create a local GPU renderer.

The worker sends actual actor/link, qpos and camera poses. The camera facade
preserves Base_Task.get_obs packing and native intrinsics/extrinsics. Reset
sends the real selected bottle model and rebuilds its service template when
the model changes. The GPU service's active visual scenes have no physics
system and share the same native context.

The idle-cloud comparison ran central first, then independent: two workers
per mode, six identical qpos actions each, initial seeds 1000000/1000001, and
reset seeds 1000003/1000004. Evidence is in `evidence/native-cloud/`.

| Check | Observed result |
| --- | --- |
| Real physical actions per mode | 12 (2 workers x 6) |
| Paired snapshots | 16 (initial + 6 actions + reset, per worker) |
| Maximum body/wrist pose, qpos, qvel, observation joint-vector difference | 0 |
| Maximum camera intrinsic/extrinsic difference | 0 |
| Paired RGB images | 64, all pixels identical; MAE 0 |
| Input actions, model IDs, action counts, success flags | Identical |
| Worker 0 bottle model across reset | 16 -> 13 |
| Worker 1 bottle model across reset | 16 -> 16, native poses changed |
| Central GPU owner | Only PID 7800; CPU PIDs 7966/7967 absent |

The wrist cameras follow actual robot links. In the aimed action frame they
show the bottle and grippers; initial-to-action wrist-image MAE is 34.37–36.33.
See `native-replay/contact-native.png` and `native-replay/comparison.json`.
This validates the sampled physical/action/observation path. No Policy was
involved, no successful task completion was established, and the short replay
is not a throughput benchmark: its logged duration includes artifact writes.

A separate two-frame entry/lifecycle check launches two independent simulator
subprocesses through `central_launch.py`. Both frames and physical states match
the earlier initial snapshots exactly, `close_env(clear_cache=True)` creates
no renderer, descriptor cleanup succeeds, and service shutdown returns the
cloud GPU to 0 MiB / 0% with no compute processes. Independent clients unregister
their attached shared-memory tracker entry; only the service owner unlinks the
RGB allocation. Evidence is in `evidence/native-entry/`.

The cloud uses Python 3.10.20, NumPy 1.26.4, Torch 2.6.0, SAPIEN 3.0.0b1,
Mplib 0.2.1, TOPPRA 0.6.3, Open3D 0.18.0 and gymnasium 1.1.1. Missing imports
were installed only into this experiment's `site/`. Source is pinned RoboTwin
`c3ddfa8b97d5519efa828b075999bd0006778e5e`; its 77 source/config files and
SHA256 manifest are retained. Three additional native metadata/SRDF files total
54,718 raw bytes, fetched from the same pinned official asset archives.
`native-environment.json` records the complete distribution inventory.

The first native smoke check used D's existing full runtime and Python 3.10
environment. Its subsequent short correctness artifacts are preserved in
`evidence/native-d/`, but other GPU jobs had appeared; those elapsed times
are excluded from performance conclusions. No other user's jobs were stopped.
The final permitted D recheck failed during SSH banner exchange, so availability
could not be established and no Policy process was started. A 9,785,132,794-byte
checkpoint file exists at `/workspace/models/starvla-robotwin-native/checkpoints/steps_40000_pytorch_model.pt`.
The idle cloud has no verified checkpoint, and weights were not transferred.
At the stage 3 handoff, fixed K2/G1 real Policy evaluation remained open. The
subsequent cloud closed-loop work above completes that experiment; the cloud
now holds the same verified checkpoint and exact D metadata.

### Replay the native correctness pair

The existing cloud directory is `/root/autodl-tmp/roboscope-central-render-20261111`.
The bundle reuses complete existing aloha/bottle assets; it does not include
model weights or unrelated object meshes. For a new minimal cloud staging,
`fetch_replay_assets.py`, `fetch_native_metadata.py`, then `fetch_native_source.py`
fetch only the pinned dependencies; the last script links native-src/assets
to RoboTwin/assets. Full native RoboTwin source/assets can be used directly.

```bash
base=/root/autodl-tmp/roboscope-central-render-20261111
export ROBOTWIN_RENDER_EXPERIMENT_ROOT="$base"
export ROBOTWIN_RENDER_PYTHON=/root/miniconda3/envs/starVLA/bin/python
export ROBOTWIN_NATIVE_ROOT="$base/native-src"
export VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_icd_abs.json
# Choose a new directory. The wrapper refuses to overwrite results or use an occupied GPU.
bash "$base/run_native_pair.sh" "$base/results/native-replay-next"
```

For another host, use its verified Python environment and Vulkan ICD path.
Reuse a compatible environment; do not upgrade shared simulator installations.
The bundle includes the source constructors under references/ and the exact
Kit launcher under roboscope_robotwin_launcher.py. Code compilation, Ruff and
Bash syntax checks pass. Recorded measured code is preserved separately from
the later entry/metadata additions in the evidence archives.

### Native versus centralized launcher

`native_entry.py` is the actual launcher switch. With no
`ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR`, it runs native GPU rendering. With that
variable, it hides CUDA devices only inside the simulator process and attaches
remote RGB; the controller and Policy process keep their GPU visibility.
`central_launch.py` owns the service, descriptor and shared memory for the
duration of an existing evaluation command. It has passed the independent
simulator lifecycle check. The D invocation below was prepared at stage 3;
the actual completed cloud invocation is in POLICY_CLOSED_LOOP.md.

On D, when it is available, `prepare_native_manifest.py` creates a new isolated
manifest, points its launcher to native_entry.py and its simulator Python to
the existing physical environment, and preserves all frozen protocol and seed
generation values. It resolves the existing runtime assets/scanner/cache paths
without modifying `/workspace/sim/robotwin` or the installed SDK.

```bash
# First copy this bundle to an isolated directory on D. Do not run on a busy GPU.
base=/home/zhaobo/roboscope-central-render-20261111-stage3
sim_python=/home/zhaobo/robotwin-work/.venv/bin/python
"$sim_python" "$base/prepare_native_manifest.py" \
  --source /workspace/sim/robotwin/benchmark-manifest.json \
  --python "$sim_python" --output "$base/policy-runtime/benchmark-manifest.json"
```

Use the existing subset runner and Policy configuration, with fixed
`--subset-task-count 1 --subset-episodes 2 --episode-shard-size 1 --concurrency 2 --replicas 1`,
`--benchmark robotwin --suite demo_clean`, the isolated manifest, and the
existing `/workspace/models/starvla-robotwin-native` checkpoint source and
`/workspace/robotwin-prep/roboscope-policy.json` config. The verified existing
controller is `/workspace/roboscope-envs/control/bin/python`, Kit 0.8.0b31, and
the subset runner is
`/home/zhaobo/roboscope-robotwin-counters-20261107/campaign-tools/experiment_subset.py`.
The prepared commands below use MPS off in both arms, identical seed prefix,
action cadence, camera quality and episode count, and distinct output directories.
They are not a record of a completed Policy run.

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$base${PYTHONPATH:+:$PYTHONPATH}"
policy_command=(
  /workspace/roboscope-envs/control/bin/python
  /home/zhaobo/roboscope-robotwin-counters-20261107/campaign-tools/experiment_subset.py
  --benchmark robotwin --suite demo_clean
  --checkpoint-source /workspace/models/starvla-robotwin-native
  --benchmark-manifest "$base/policy-runtime/benchmark-manifest.json"
  --policy-config /workspace/robotwin-prep/roboscope-policy.json
  --subset-task-count 1 --subset-episodes 2 --episode-shard-size 1
  --concurrency 2 --replicas 1
)
env -u ROBOTWIN_CENTRAL_RENDER_DESCRIPTOR "${policy_command[@]}" \
  --output-dir "$base/policy-native-next"
"$sim_python" "$base/central_launch.py" \
  --robotwin-root /workspace/sim/robotwin/RoboTwin --scenes 2 \
  --log "$base/policy-central-service.json" -- "${policy_command[@]}" \
  --output-dir "$base/policy-central-next"
```

The pinned cloud source subset lacks the full official evaluator/seed generation
and a checkpoint; it is a physical replay environment, not a complete official
50-task runtime. Do not use these correctness results as benchmark acceptance.
The reusable code bundle is `central-render-stage3-bundle.tar.gz`. Raw states,
RGBs, logs and the measured code are preserved separately in
`evidence/native-cloud-evidence.tar.gz` and `evidence/native-entry-evidence.tar.gz`.
The original central result JSON is 565,110,290 bytes because the camera facade
returned its internal metadata dictionary and native get_obs added RGB to it
in place; the artifact logger then repeated those pixels. The facade now returns
fresh metadata, matching the native camera contract. The two entry frames still
match the prior states/RGBs, and metadata contains no RGB arrays. The bundle
contains `result-summary.json` (180,541 bytes for central), a small alignment
image, and links to the preserved originals, rather than that large JSON.

## Run the stage 1 equal-work pair

Use an existing Linux RoboTwin environment and its assets. No model checkpoint is
needed. Copy these Python files and `run_pair.sh` to an isolated directory on the
selected host. Read any applicable host/repository AGENTS.md before writing there.

```bash
# Native D paths already verified to exist. Run only when its GPU is free.
# The environment variables match its existing CUDA/Vulkan installation.
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

bash run_pair.sh \
  /home/zhaobo/robotwin-work/.venv/bin/python \
  /home/zhaobo/robotwin-work/RoboTwin \
  /path/to/isolated/kit/benchmarks/robotwin/roboscope_robotwin_launcher.py \
  /path/to/isolated/results/n2 2 100
```

`run_pair.sh` checks for other compute processes and GPU activity before each arm;
it stops without modifying them if the GPU is occupied. An environment probe
imports packages but creates no renderer. Installations are never modified.

The default workload uses `adjust_bottle`, `demo_clean`, scene seeds 1000000 and
1000001, all native cameras, RT 32 samples/pixel, path depth 8 and OIDN. D435 is
320x240 in the pinned camera YAML. The actual source/config hashes and output
shapes are recorded by the script, rather than assuming that a local asset copy
is unchanged. The camera resolution and renderer quality are never overridden.

The current Kit main launcher has no front-camera exclusion. The default keeps
front_camera, where present, in both arms. To measure the explicit three-camera
variant later, set `CAMERA_SET=policy` for the entire pair. Never compare an
all-camera arm with a three-camera arm.

Each scene receives one separately recorded first observation, five warmups,
then exactly 100 timed observations. Initialization, images written to disk and
teardown are excluded. Independent mode spawns N processes with one scene each;
shared mode spawns one process with N distinct scenes. The shared engine,
renderer and native `_internal_context` are asserted to be identical, while the
scene render systems are distinct. Images are submitted for the active request
group before CPU RGB retrieval. A `--schedule scene` option supports a targeted
follow-up with sequential submission if necessary.

The timed path is `_update_render`, native camera `take_picture`, Color readback
and uint8 conversion. There is no timed physics, policy inference, full
`get_obs` assembly or IPC. Request completion latency starts with a synthetic
simultaneously ready request group, including its queue position. This is not a
measured worker-to-service network round trip. Scene order rotates per frame.

Results include:

- `environment.json`, `independent.json`, `shared.json` and per-arm logs.
- Exact total images and aggregate images/s over earliest-start to latest-finish.
- Batch and request-completion mean/p50/p95, first-render and setup times.
- Device-wide VRAM snapshots and approximately 250 ms samples (including any
  baseline display/service memory); these are not per-process memory shares.
- Per-scene warm/last RGB NPZ files, shapes, image statistics and hashes.
- Scene entity poses, camera intrinsics/extrinsics and source hashes. The
  comparator rejects mismatched work, states, cameras, shapes or source hashes.
- Cross-arm image MAE/p99/exact fractions. RT stochastic differences require
  inspection; the comparator does not equate different images merely because
  their dimensions match.

Start with N=2. Expand to N=4 only if the result or capacity question needs it.
N=12 is conditional on both arms fitting with identical settings; no memory
estimate is used as a measured capacity result.

## Current-version SAPIEN reuse

The installed native environment reports SAPIEN 3.0.0b1. Its `sapien.core` has
neither `RenderServer` nor `RenderClient`; `sapien.render` provides
`RenderSystemGroup` and `RenderCameraGroup`. The 2.2 server example is therefore
not a drop-in backend for this pinned runtime. Downgrading SAPIEN would change
the simulator and is outside this experiment.

The version-pinned primary sources are saved in `evidence/sources/` with URLs
and SHA256 values in `manifest.json`:

- `sapien_renderer_system.cpp`, `SapienRenderEngine::Get`:
  reuses a live process-wide engine, context and resource manager. Each scene
  has its own RenderSystem; a single Python Engine wrapper alone is insufficient
  evidence of sharing.
- `camera_component.cpp`, camera initialization: camera creation selects the configured
  shader and applies spp, path depth and OIDN to the RT renderer.
- `batched_render_system.cpp`, group construction/render: camera groups require equal
  dimensions, initialize the existing cameras, call each existing renderer and
  copy named render targets to CUDA buffers. This code contains no raster-only
  rejection or shader downgrade. It retains the previously configured RT
  renderer, but does not merge rays into one render invocation.
- `python/pybind/sapien_renderer.cpp`, group bindings: group creation, picture
  submission and CUDA picture access are exposed. There is no old server RPC API.

Consequently a minimal native fallback can keep multiple visual scenes inside a
single process and return their Color observations over a RoboTwin-specific IPC
protocol. Reuse the current per-camera RT/OIDN path first; an optional camera
group can batch output copies on a defined CUDA stream. Actual RT/OIDN execution
and output equivalence through the group remain untested in this stage; source
compatibility is not a performance result. `RenderSystemGroup` does not implement
remote scene creation, camera/state serialization, a request queue, or shared
memory responses.

The native fallback is implemented by stages 2 and 3 above. Physics-only workers
also avoid material/camera code that implicitly creates a GPU renderer;
removing take_picture alone would not establish CPU-only rendering ownership.

## Stage 1 validation and scope

Python compilation, Ruff, CLI parsing and Bash syntax pass. The actual cloud
replay completed both arms, context/state/source assertions and pixel checks.
The native full-task path was prepared on C but was not benchmarked because its
GPU was busy. `evidence/discovery.json` preserves the admission observations;
they are not performance results. No existing task or service was stopped.
No 2x/4x claim, IPC measurement or full rollout result is reported for stage 1.
