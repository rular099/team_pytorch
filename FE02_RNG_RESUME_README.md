# FE02 已有checkpoint的RNG续跑修复

将此小补丁包解压到**原超算代码目录**
`/public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_fe02`。
包只新增本说明、`resume_fe02_rng_cpu.sh` 和 `compat/fe02_rng_cpu/sitecustomize.py`；
不包含/替换原训练源码、SOURCE_IDENTITY.json、audit lock或任何checkpoint。

在该目录直接运行：

```bash
bash resume_fe02_rng_cpu.sh
```

默认直接提交M/seed42（index4）从last续跑。原有配置、4节点×4卡、输出根保持不变；
无需重训、重跑audit、修改权重、重装PyTorch或在登录节点加载torch。
预览可选`--dry-run`；其他已失败组需显式`FE02_INDICES=...`。

原因：原恢复代码torch.load(map_location=device)将保存的CUDA RNG状态ByteTensor
也映射到DCU；PyTorch1.13的set_rng_state_all要求CPU ByteTensor。
此层仅在明确启用的训练rank中，将传给该函数的状态张量搬回CPU，不改变uint8字节、
顺序或seed，不跳过恢复。原源码/配置/data/weights/world身份检查仍原样执行。
兼容层独立SHA由提交环境快照固定，各rank启动再次验证；成功恢复后向原输出根
`compatibility_receipts/`写每rank记录，日志出现`[FE02-RNG-CPU]`。
登录节点与verify_source/select_run等非rank辅助进程不因此导入torch。

本地测试只验证字节保持、启用范围、校验/失败阻断及原code_identity不变；
实际超算续训仍需以新任务日志和checkpoint epoch验证。
