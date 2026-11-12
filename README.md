# RoboTwin Central Render

RoboTwin 每 GPU 一个渲染服务的实验实现与 Claude 交接，2026-11-13。

**K2/G1 真实 Policy 闭环已完成；当前没有证明端到端加速。**
两种模式动作序列相同、均成功 2/2 集；总墙钟 29.538→29.632 秒，基本不变。
正式窗口吞吐 34.255→41.351 步/s（+20.718%），但首次推理冷热差异显著；
后续较暖的原生运行达到 38.795 步/s。峰值显存 16708→13956 MiB（−16.47%）。

- **接续入口：[CLAUDE_HANDOFF.md](CLAUDE_HANDOFF.md)**：架构、实验结果、远程环境、已解决问题及下一步。
- [POLICY_CLOSED_LOOP.md](POLICY_CLOSED_LOOP.md)：最新完整闭环报告和精确计时边界。
- [EXPERIMENTS.md](EXPERIMENTS.md)：阶段 1～3 和后续闭环的历史记录。
- [closed-loop-result.json](closed-loop-result.json)：机器可读最新结果。
- [evidence/policy-cloud/closed-loop](evidence/policy-cloud/closed-loop)：所有完成及失败闭环运行。
- [THIRD_PARTY.md](THIRD_PARTY.md)：上游版本和许可。
- [EXPORT_NOTES.json](EXPORT_NOTES.json)、[EXPORT_MANIFEST.json](EXPORT_MANIFEST.json)：合并导出说明和文件校验。

代码按原始布局直接展开，支持修改和 git diff。模型权重、完整资产和环境不放入 Git；
现有云主机已准备这些依赖，位置和复跑命令见交接文档。SSH 密码由用户单独提供。

旧 stage3-result.json 记录的是 Policy 尚未运行时的阶段状态，不能代替最新闭环报告。
“48 worker 最多 2 倍、100+ worker 最多 4 倍、5000 集估时”未在本项目验证。
