# A01 数组30265280：恢复错误与启动超时

仓库 `rular099/team_pytorch`，分支 `exp/fe01-a01-absolute-amplitude-ablation`，
修复基线 `fdab9e23c7dd73066f4d2032f23ab99d8f9190b6`。
输入是用户提供的 `chaosuan_res/dcu_continue_20261011T094505_AiltUR.zip`，4份out/err，
SHA256 `6dbab6c481f3d07309288b68327d4f7bac81e482d4fa6cae35ea5db477dab326`。
原文件未修改；[逐成员SHA与提取事实](log_evidence.json)。

| 索引 | 任务 | 本批通信探针 | 本批训练结果 |
|---|---|---|---|
| 3 | DiTing ON | 16/16 PASS，sum136 | 打印从epoch5恢复；16个rank报RNG ByteTensor错误，无新epoch |
| 4 | DiTing OFF | 16/16 PASS，sum136 | 打印fresh；16个rank触发1200秒启动超时，无新epoch |

## 已证实的错误与仍未知的问题

ON traceback精确指向旧核心的 `torch.cuda.set_rng_state_all(rng['cuda_rng'])`。
同函数先用 `torch.load(..., map_location=device)` 读取last.pth，
导致原本为CPU ByteTensor的cuda_rng也被映射到DCU。CPU rng已有 `.cpu()`，
而DCU RNG列表没有。PyTorch1.13.1的
[官方实现](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/random.py#L39-L66)
克隆传入张量并交给generator，不会自动迁回CPU；本地合成设备载体与真实ByteTensor
generator API复现了恢复前失败、转换后状态及后续随机抽样一致。
本地真实CUDA/ROCm回归因无设备而跳过，不能声称已通过超算恢复。

ON的恢复标记出现在旧checkpoint元数据检查和备份之后，因此日志支持这次检查了epoch5。
本ZIP不含checkpoint本体、目录清单、执行源码SHA或sacct；不能独立重算这些SHA或断言
当前远端last仍是同一文件。新任务继续在计算节点认证实际last。

OFF没有Python traceback，也没有本批RSMI mutex/rsmi_init失败证据。
最新这次使用a02r1n18/a02r1n19/a02r2n00/a02r2n01，和之前RSMI错误那批不同。
20分钟没有新的非空journal，只证明触发启动看门狗，尚不能区分慢初始化、DDP/RCCL阻塞、
数据加载或首批计算等情况。不会把之前的RSMI错误直接当成本次根因，也不凭这些日志
继续扩大节点黑名单。补充阶段和调用栈以获得下一步依据，暂不放宽20分钟上限。

## 已实施的恢复路径

[新提交入口](../../scripts/fe01_a01_ops/submit_diting_resume_fix.sh)默认仅3,4，仍4节点16卡/项。
[操作说明](../../docs/ai/FE01_A01_RESUME_FIX.md)明确本地上传、登录节点提交和计算节点行为。

- 独立静态循环只增加观察语句和RNG状态CPU转换；原Adam/调度器、采样、DDP、
  损失、12轮2544updates、世界大小、初始化、validation选checkpoint规则均保留。
  不通过全局monkeypatch替换原函数，不删除或重签旧audit/门控。
- 启动时用AST检查允许差异，实际全局依赖按对象身份核对。
  新执行循环与运营文件的SHA记录在独立execution_identity，随返回包保存。
  原核心SHA是认证协议来源，不能单独替代新执行身份。
- ON原last/best先备份，恢复模型、Adam、schedule、Python/NumPy/Torch/各设备RNG，
  不重新从头训练已提交5轮。实际恢复起点由检查通过的last确定。
- 若OFF只残留无任何提交/采样进展的初始化目录，在锁内完整移动到operation存档，
  再重新初始化；有best、临时checkpoint、曲线、非空journal或未知文件则停止。
  不清理旧目录、不丢弃已提交训练，不把ON纳入此fresh重启例外。
- 逐rank记录最后阶段和120秒周期栈；20分钟上限保留。
  新确认的第一次optimizer update也可解除启动限制，避免Python journal缓冲未落盘时误判。
- 普通退出时自动打包每索引小型return_bundle，不携带权重、环境私有内容。
  bundle中的batch exit code与Slurm最终sacct状态分开说明。

本地聚焦测试、Python3.9语法、bash语法、原115文件/A01核心SHA、三个原工作树保护、
新上传包的假调度器验证见 [verification.json](verification.json)。
**本轮未提交真实sbatch/srun，未运行正式训练/生产评价，未访问held-out test。**
