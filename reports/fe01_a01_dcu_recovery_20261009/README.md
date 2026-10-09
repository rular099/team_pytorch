# A01 train 的 DCU 启动失败处理

仓库 `rular099/team_pytorch`，分支 `exp/fe01-a01-absolute-amplitude-ablation`；
修改前HEAD为 `3ecf060a3334d1c7b48662cf4395a3bab07723ff`。

用户报告本批train 2项成功、3项失败，提供了
`AssertionError: DCU unavailable in allocation`。尚无本批完整out/err、JobID和索引，
不能认证两项成功的最终epoch，不能定位具体故障节点，也不能认定驱动坏了。
本地原有20261007失败压缩包属于旧批次，不能用于补造这次节点或作业信息。

已核验源码：原 `job.sbatch` 先source `env.sh`，该断言在 `env.sh` 的torch检查，
发生在训练srun之前。因此本次这些失败尝试尚未进入训练入口。Slurm的GRES/可见性
作用域见 [官方说明](https://slurm.schedmd.com/gres.html)；具体集群是否为batch/step
差异、节点权限或HIP初始化问题仍需真实设备记录。

交付的独立 [dcu_recovery.sh](../../scripts/fe01_a01_ops/dcu_recovery.sh) 有三个用户入口：

- `inspect`：只读最近train数组的stderr，自动识别这个启动断言；不猜测其余项成功。
- `probe`：显式提交4节点16rank、10分钟的设备诊断，对比batch与实际step；不读权重/数据。
- `retry failed`：自动识别旧启动失败项，逐项执行原preflight；全部通过才提交一个数组。
  也可显式用 `retry 0,2,4`。已有非空训练目录被拒绝，成功任务不会覆盖。

重提在实际step内对所有rank检查原环境和设备运算，然后运行未改动的原训练/缓存启动器。
保持原资源、world16、模型、loss与预算。不删除设备可见性限制、不回退CPU、不重签lock。
启动器及私有设置的只读副本、各自SHA、原源码manifest和提交参数另存operations目录，
使部署操作与科学源码来源分别可追踪。旧核心SHA、五项门控与成功输出无需改变。
其效果是修正启动检查位置并提供实际故障信息，尚不能声称已修复不可访问的节点。

本地检查：23 passed、1个timm弃用提示，11.72秒。包含9项新操作测试，以及原调度器、
后端与身份检查。假环境验证batch失败/step成功、step失败阻止训练、原初始化异常写JSON、
零误提交、只提交失败索引、完成输出拒绝覆盖、门控与私有设置未变。
测试没有真的调用Slurm或设备；`verification.json` 记录语法与兼容性核验。

生产step probe为NOT_RUN，新超算作业为NOT_SUBMITTED。没有自动训练、评价或test访问。
使用方法和输出位置见 [操作说明](../../docs/ai/FE01_A01_DCU_RECOVERY.md)。
