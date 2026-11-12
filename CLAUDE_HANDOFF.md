# RoboTwin 集中渲染：Claude 接续说明

交接日期：2026-11-13。实验完成日期：2026-11-11～12。本次交接没有新增 GPU 实验。

## 目标与已实现架构

继续已有实现，验证“每张 GPU 一个渲染服务，多个 CPU 物理 worker 共享”的实际端到端收益。不是从零实现通用 Vulkan MPS。

K 表示仿真 worker 数；G 表示独立 Policy 推理进程数。本次完整闭环是单 GPU **K2/G1**，MPS 关闭。CPU worker 保留 RoboTwin 原生物理和控制，将物体、机器人 link、相机位姿通过本机 Unix IPC 发给服务；服务为 worker 保留独立视觉 scene、共享进程内原生 Vulkan context，通过共享内存返回 RGB。请求带 worker/scene/episode/generation/frame 标识。

已验证范围：adjust_bottle / demo_clean / aloha-agilex / qpos。不是全部 RoboTwin 任务通用实现。SAPIEN 3.0.0b1；四个 D435 相机，320×240，RT32 SPP、path depth 8、OIDN；front_camera 仍渲染。Policy 使用 head/left/right，动作 horizon 与观测 stride 均为 16；没有视频、深度或点云。

## 先读这些文件

1. `POLICY_CLOSED_LOOP.md`：最新闭环结果、计时边界、运行修复及复跑命令。
2. `closed-loop-result.json`、`closed-loop-provenance.json`：机器可读结果与版本/模型哈希。
3. `EXPERIMENTS.md`：阶段 1～3 及闭环的完整历史说明。
4. `native_backend.py`、`native_entry.py`：原生物理接入、渲染切换、相机 facade。
5. `central_render_service.py`、`robotwin_worker_adapter.py`、`visual_scene.py`：集中渲染、IPC、共享内存与 scene 生命周期。
6. `policy_run.py`、`policy_trace.py`、`run_policy_arm.sh`、`analyze_policy_pair.py`：闭环执行、统一计时、结果分析。
7. `evidence/policy-cloud/closed-loop/`：保留所有成功、失败及后续敏感性运行的原始记录。

`stage3-result.json` 是历史文件，当时尚未跑 Policy；不能用它判断当前状态。旧 bundle manifest 是原始打包时的哈希，合并导出清单是 `EXPORT_MANIFEST.json`，导出差异见 `EXPORT_NOTES.json`。

## 已完成实验

| 阶段 | 原生独立渲染 | 集中渲染 | 能说明什么 |
| --- | ---: | ---: | --- |
| 固定视觉回放，N2，800 张图 | 75.0661 图/s | 93.9357 图/s | +25.14%；16 张配对样本逐像素相同；无 IPC、无 rollout |
| 动态合成位姿 + IPC，N2，800 张图 | 65.4815 图/s | 78.5238 图/s | +19.92%；32 张运动/reset 样本逐像素相同；不是物理闭环 |
| 原生物理、2 workers、每种模式 12 次真实 qpos 动作 | 16 个状态快照、64 张 RGB 配对比较 | 全部差异为 0 | 覆盖模型 16→13 reset、腕部相机跟随；不能作为稳定吞吐测量 |

回放显存约 6480→3727 MiB。固定回放中的机器人处于 nominal pose，腕部视野主要是墙，不能外推复杂场景。

真实 Policy 闭环的所有完成运行如下，按发生顺序列出：

| 指标 | native-k2g1-r2 | central-k2g1-r3 | 后续 native-k2g1-r3 |
| --- | ---: | ---: | ---: |
| 完成/成功集数 | 2/2 | 2/2 | 2/2 |
| 两个 seed 的动作步数 | 110 / 124 | 110 / 124 | 110 / 124 |
| 总动作步数 | 234 | 234 | 234 |
| Policy forward 次数 | 15 | 15 | 15 |
| 观测数 / RGB 张数 | 17 / 68 | 17 / 68 | 17 / 68 |
| 对齐正式窗口，秒 | 6.831210 | 5.658804 | 6.031630 |
| 正式窗口，步/s | 34.254549 | 41.351497 | 38.795482 |
| 含加载/启动/退出的总墙钟，秒 | 29.538470 | 29.631902 | 28.613666 |
| 第一次推理，秒 | 1.970766 | 0.546450 | 0.628985 |
| 后续 14 次推理均值，毫秒 | 59.051 | 63.966 | 63.340 |
| 峰值 GPU 显存，MiB | 16708 | 13956 | 16708 |

三次运行的完整动作序列相同，均按成功终止，step limit=400。真实进程记录确认 2 个仿真进程、1 个 Policy 进程；集中模式的 CPU worker 没有 GPU 渲染上下文。

**当前结论：功能闭环已跑通，尚未证明端到端加速。**

- 第一对正式窗口吞吐 +20.718%，窗口耗时减少 17.162%。
- 总墙钟 29.538→29.632 秒，增加约 0.316%，基本不变，不能声称端到端提升 20.7%。
- 首次推理差 1.424 秒，比正式窗口总节省的 1.172 秒还大。存在明显冷热/启动混杂；不能把正式窗口收益直接归因于渲染。
- 与后续较暖 native 相比，集中模式正式吞吐约 +6.59%；这不是成对重复验证，只是敏感性观察。每次仅 2 集、约 6 秒正式窗口，不能宣称稳定收益。
- 峰值显存减少 2752 MiB，约 16.47%，这是本次实际观察。
- 后续 native 测试是在收到停止指令前已启动并完成的；保留结果避免选择性汇报，没有继续补跑 central。

正式窗口定义：两个 worker 的首个真实观测都准备好，且都尚未调用首次 Policy 时统一释放 barrier，到最后一个 episode-end callback。首次推理仍在窗口内。总墙钟从 CLI/服务启动前计时，到完整退出；包含加载和初始化。SDK 自带计时另有边界，集中服务启动在 SDK 之前，不能拿 SDK 的 12.8587→19.4098 步/s 当公平的端到端提升。

## 原始 idea 的背景与证据边界

MetaWorld K3/G3 旧实验：107.20→140.73 步/s（+31.28%），2500 集总耗时 61.13→46.73 分钟，成功率均 75.60%。这是多 CUDA Policy 进程能够受益于 MPS 的实际例子；MPS 不直接加速 Vulkan/OpenGL 渲染。

RoboTwin 历史 G2 对照也没有显著收益：K2/G2 20 集 207.849→207.299 秒（约 +0.27%）；K4/G2 100 集 543.577→538.876 秒（约 +0.87%）；K6/G2 100 集 409.809→407.248 秒（约 +0.63%）。固定 K6，G1→G2、MPS 关闭，100 集 414.535→414.321 秒（约 +0.05%），显存多约 10.056 GiB。因此不能简单归因为“之前只有 G1”，也不能说 K:G=1:1 就一定有效。

这些历史数字来自前序记录，配置与当前集中渲染实验不同。参考上游报告：
https://github.com/QLL-CCDD/RoboScope-Kit/blob/572ff00cff5fa398b2fe6818d22ee7bd7676f8e6/benchmarks/robotwin/profiling-native-mps.md

K1 串行诊断中 action 62.26%、obs 30.09%、Policy RPC 7.53%，不是 48 worker profile。已有证据支持仿真/观测/同步限制了整体推进，但没有单独证明 Vulkan context 切换是唯一原因，也不能把低 RT 指标等同于没有渲染瓶颈。

用户旧表格“48 worker、跳过 front_camera、583 步/s、5000 集 3800～3900 秒，集中渲染最多 2 倍、100+ worker 最多 4 倍”是另一配置下的基线转述和预测。本仓库没有验证 48/100 worker 或 5000 集，且当前四相机保留 front，不能直接横向比较；30×/50×官方也未验证。

## 远程算力与已有环境

```bash
ssh -p 42771 root@connect.westd.seetacloud.com
```

密码由用户在聊天中单独提供，未写入仓库。最后实测日期 2026-11-12；连接与 GPU 是否仍空闲需实际检查，不能把历史状态当实时状态。不要终止其他作业。

- 根目录：`/root/autodl-tmp/roboscope-central-render-20261111`
- GPU：驱动报告 RTX4090、49140 MiB；驱动 580.105.08，主机 `autodl-pro-79387867b0ad`。
- Policy/仿真 Python：`/root/miniconda3/envs/starVLA/bin/python`，3.10.20，Torch 2.6.0+cu124。
- 控制 Python：`$base/policy-env/bin/python`，3.11.15；SDK 0.8.0b31，`sdk-site/`、`site/` 已准备。
- Vulkan：`VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_icd_abs.json`。
- StarVLA：`/root/starVLA`，commit `7e4a47360093d79dc42b883885c04fbe790c5b60`。
- RoboTwin：官方 `c3ddfa8b97d5519efa828b075999bd0006778e5e` + 确定性 annotation patch，运行时 commit `0fe7d15a86ae19d9411a52520683039cfc81d9ff`。
- 权重已经下载，路径 `$base/models/starvla-robotwin-native/checkpoints/steps_40000_pytorch_model.pt`，9,785,132,794 字节。
- 权重 SHA256：`8ccee8ddb33fa5df83a9a5e2b7d4f737db93f19a051b23f738af8a484fe2d7e5`。
- Hugging Face：`StarVLA/Qwen3-VL-OFT-Robotwin2`，revision `e9cdfc38a0db968bf3a865bcb542d3d455952226`。
- 已有 `policy-runtime/`、`policy-source/`、RoboTwin 资产、seed cache、模型元数据、原始 `closed-loop/` 结果。优先复用，不重新下载 10 GB 权重或重装 CUDA/物理栈。

仓库是源码和证据交接，不包含这些大体积依赖。直接 clone 不会自动变成可运行的完整环境。先读现成远程目录和脚本路径；在新 checkout 上复用/链接既有依赖或明确修改路径，不要覆盖旧实验目录。`stage_public_runtime.py`、`fetch_replay_assets.py`、`fetch_policy_metadata.py`、`bootstrap_control.py` 提供相应准备过程，`frozen-inputs/`、`model-metadata/` 和 wheel 固定输入。

已有固定 K2/G1 命令如下（先阅读脚本，换新输出名）：

```bash
base=/root/autodl-tmp/roboscope-central-render-20261111
bash "$base/run_policy_arm.sh" native new-native-name
bash "$base/run_policy_arm.sh" central new-central-name
"$base/policy-env/bin/python" "$base/analyze_policy_pair.py" \
  "$base/closed-loop/new-native-name" "$base/closed-loop/new-central-name" \
  --output "$base/new-comparison.json"
```

脚本当前固定 1 task、2 episodes、K2/G1；这只是原实验复跑入口，不是下一轮长稳态测试已经实现。需要调整 episode 数或预热时，必须同步两种模式，并清楚记录改动。

## 已解决的问题与仍可能遇到的问题

已解决：

- `python -I` 子进程忽略 PYTHONPATH，用私有 SDK site 配置修复。
- 物理 URDF cache 并发写，改为临时文件 + 原子 rename。
- 官方 `Sapien_TEST` 在 CPU-only worker 中偷偷创建本地 GPU renderer 导致崩溃；集中模式将这项 GPU preflight 交给已就绪服务。任务物理与相机配置未删减。
- `get_obs` 会向相机 config 写 RGB；facade 返回共享内部字典曾造成日志不断膨胀，`get_config` 改为独立 metadata copy。原先约 565 MB 日志只导出既有 summary，避免塞入仓库。

需要继续关注、但不应先展开大矩阵测试：

- 冷启动、首推理、reset 和 steady-state 的边界是否一致。
- 多 worker 时单渲染服务是否排队饱和；IPC 等待、RGB 复制、OIDN/CUDA 同步是否吞掉收益。已有残差 44～47 ms 包含排队与复制，不能称为纯 IPC 开销。
- 新任务/随机化下 geometry、材质、光照、机器人与物体变化的镜像完整性；现有结果只覆盖 adjust_bottle 和已测 reset。
- SAPIEN 3.0.0b1 没有旧 2.2 的 RenderServer/RenderClient。别按旧 API 从头做通用 Vulkan 拦截器。
- 当前服务共享 context 不等于任意渲染都自动批处理/并发，调度仍可能串行。

## 接下来怎么做

1. 先读取证据与代码，确认远程环境、已有模型和当前 GPU 占用，避免重复准备。
2. 最小有效下一步：同 GPU、同 K2/G1、同 seed/任务/四相机质量，对两模式使用一致的模型预热与计时边界，延长到有代表性的正式窗口。分别报告含加载总墙钟、正式吞吐、成功/动作步数、显存；不要通过直接减去重叠 span 制造“校正速度”。
3. 根据这对闭环和现有时间记录决定是否优化排队/拷贝/同步。只有集中模式确实改善整体推进，再做一个小幅 K 扩展（如 K4 或 K8，固定 G1）确认趋势；不要直接跑 48/100 worker 或全任务矩阵。
4. 每轮实验前概括目的、变量和预计工作量，结束后概括现象与真实端到端提升。如果卡住或耗时异常，判断当前步骤是否必要，取消无必要步骤。快速实验，不堆叠防御性实验。

保持 native/central 可切换，保留失败记录。若性能没有改善，明确报告；不要把图片回放加速、SDK 窗口偏差或理论估计包装成真实 rollout 收益。
