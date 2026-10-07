# FE01-A01 实施与验收边界

任务：`20261007-fe01-a01-geometry-absolute-amplitude`。
基线：`ba4fa7740d9fde5fde87a7b0ae0397037209baf4`。
分支：`exp/fe01-a01-absolute-amplitude-ablation`。独立 worktree，原三个工作区保持不动。

## 模型与训练

`A01FullModel` 继承 FE01FullModel，只覆盖交付给幅值投影的输入策略。
显式 factory 注入该类；没有全局 monkeypatch。ON 的11维不变；OFF前10维零、末维时长不变。
公共参数、adapter、gate、状态名称/形状、归一化、窗口、组件比例和辅助任务不变。
配置和 checkpoint 都保存 `absolute_amplitude_mode`。
校验器拒绝 raw-energy/token weighting、DPK、temporal、anchor/RT60/RT61 旁路。
没有把 RT61 标为无振幅，也没有更改其加载器。

训练循环是基线 FE01 循环的独立静态副本，保留 Adam、loss、microstep平均、
梯度裁剪、DDP no_sync、drop-last、cosine、验证选择与严格更好才换 best 的行为。
新增的部分是 A01 factory、已认证 epoch0 初始状态恢复和独立来源字段。
保留实际 world16；相同 global batch 本身不能证明 loss 分母一致。
sampling manifest 固定 event/epoch/draw/time，DDP order 另存每 epoch/rank 的消费索引指纹。
DiTing 两组还要匹配完整初始状态与 sampler manifest。

旧 ON 复用门控核对冻结文件 SHA、epoch/config/lock/world、真实预训练依赖、
完整 frontend/adapter/common 初始状态、新旧 ON 前向与固定 mini-batch 更新。
TEAM 只有实际 scratch 且预训练 SHA 为 null 才通过；pretrained 必须合法非空 SHA 并核对文件。
若 init 缺失，重建后必须匹配已下载冻结 inventory 的完整状态与 common 指纹。
OFF 从 epoch0 开始，绝不从 ON 最终 checkpoint 微调。

## 查询和幅值诊断

探针按 time、geometry、实际 K1/Kmulti、Q1–4/Q≥5 分层，每层最多32 decisions。
选择只用 metadata 哈希与数量，不用预测误差、PGA值或模型表现。
逐层 hook 保存 PE、attention 前 query、pure attention/weights、residual sum、
norm 后输出、每层 readout、MLP 与最终 MDN。输入坐标另存。
实际 scalar gate、分支范数、均值与 weights/mu/sigma 的跨 query 变化分别导出。
query eval 路径显式 enable_grad，无 outer no_grad、detach 缓存或 optimizer；
纬经度按绝对纬度转换为公里，有限差分步长0.01/0.1/1km；高程单位未认证，不宣称其公里导数。
station 同步置换用于等价控制，坐标单独置换是机制干预，不记真实评分。
嵌套 K 固定 event/T 与共同目标，按 pick+station ID 排序，排除最大输入集合；不足时报告 unsupported。

前缀测试固定标签、元数据和请求，从 exclusive cutoff 开始扰动 pulse/NaN/Inf。
比较选站、mask、去均值/方差、归一化 peak、10维幅值、duration、encoder/adapter、MDN、概率。
触发/容量/缺测/合法零/invalid station 与最后合法样本正控制都有入口。
正控制仅要求相关统计变化，不要求任意模型预测必变。
逐站正增益和事件共同增益控制在正常数值区间核对 shape、OFF features/MDN、
ON log10平移和时长；极低能量 eps/clamp 单列限制。
HDF前缀检查不覆盖源头离线滤波/重采样：`UPSTREAM_CAUSALITY_UNKNOWN`。

## 评价与来源

只有 validation 1/3/5/10/20s × normal/random 共同十cell。
必须通过194265 all、139440 noninput、1310 events及原 key/label/input/clock SHA。
发现缺失/额外请求或物理标签变化即输出差异并失败，不取交集。
原 CSV 二次序列化存在 float64显示舍入：先严格核对实际 float32标签与整数时钟，
再使用原冻结 CSV 的十进制显示作为身份表示，不重签旧 lock或修改旧115文件。
精确 NLL从原 logits 的 log-softmax计算，避免先 float32 softmax 的下溢。
旧 ON 评价重新使用冻结 epoch 的相同权重，不重新挑 epoch；同时报告 epoch12。
误差、精确 MDN 分布指标、level/shape、动态范围和近等距离对联合报告。
bootstrap 用 event为簇，稀疏子群列缺 cell、实际数与有效重复，不冒充完整十cell结论。

原115文件 SHA与 A01新增源码 SHA 分开。包内独立 manifest 校验实际文件。
新源码版本使 identity/diagnostics/pilot gate 失效。
本轮不提交超算作业，不下载权重、不访问 held-out test、不改 EVAL1 TEAM gate。

可执行入口见 [HPC 操作说明](FE01_A01_HPC_RUNBOOK.md)。
实测输出与解释见 `reports/fe01_a01_local_20261007/REPORT.md`。
