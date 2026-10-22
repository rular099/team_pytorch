请按用户给出的精确result commit复核FE01 ON V4。

仓库rular099/team_pytorch；分支exp/fe01-on-warning-map-evaluation；基点29ffb20ba0effc0c345a0329ba3f8c658dc8cef0。
先回报实际读取的repo/branch/commit，再读AGENTS.md、docs/ai/PROJECT_CONTEXT.md、docs/ai/README.md及本目录README.md。

研究规范是configs/fe01_a01_compare中的完整V4任务书/研究合同。
实现是fe01_a01_compare、scripts/fe01_a01_compare；实际本地验证见docs/ai/FE01_ON_EVAL_VERIFICATION.md和本目录verification。
正式超算NOT_SUBMITTED，真实生产checkpoint/模型smoke/逐秒/grid/时延NOT_RUN；synthetic只能作功能证据。

请分别审查：

1. 既有四ON跨模型结果：原十cell noninput、第一秒、早期1/3/5秒的配对差和区间，selected/epoch12、all/K1/untriggered/空间场的不利和不确定结果。核事件权重、RMSE开方位置、pooled尾部与R²/slope定义。不要把单seed事件CI当重训CI。
2. 固定分桶/四交互及共同time/K支持域：完整Unknown/稀疏支持、有效draw、<20事件描述性；SNR不可见时不应出现低SNR优势结论。支持先冻结、差值方向、四格联合抽样及reference水平是否正确。
3. 身份兼容：旧115/A01源不变、TEAM训练encoder、实际state/文件SHA、原pretrained与source/revision门、round_trip；新release身份与原训练身份分开。不得以脱敏JSON的SHA差篡改源seal。
4. 新固定query/网格回放是否隔离未来数据、labels和目录源信息；任意query、顺序/分块/追加grid、同cutoff缓存及跨cutoff/IDs/mask/ckpt拒绝；模型级证据仍待HPC，不因存在smoke代码而判已通过。
5. SNR/缺测/UTC与真实首次越阈的物理分量、滤波/单位、左删失/缺测/clipping/峰值不一致状态；完整目标波形只能由离线truth模块使用。
6. TTUM的N>=5固定面板、首次/三帧稳定起点/确认、尾部/首帧删失、全分母；shadow的lead=0、horizon未知、无input/丢文件、空H和误报。没有独立共同空间支撑应为面积NA。
7. 数值/渲染交付、同步计时、cold/warm和FIFO积压；通信未知，不能把离线批吞吐或query均摊时间叫部署时延。图源/格点高程估算/缺测/统一色标、全部事件索引与12个metadata预选案例是否诚实。
8. Bash/Slurm任务依赖、错误恢复、原子发布/完整seal、UNKNOWN不算成功，不重复在跑项；prepare/status不加载模型、不提交。检查是否存在会导致真实HPC失败的实现问题，给文件/行号/严重性。

最后区分“本地已验证事实 / 基于代码的推断 / 必须由HPC返回的证据”。不预设DiTing必胜，不默认要求重训或打开test。
对照研究边界判断后续最小证据即可，不以较低MAE直接断言盲区或成图时间改善。
