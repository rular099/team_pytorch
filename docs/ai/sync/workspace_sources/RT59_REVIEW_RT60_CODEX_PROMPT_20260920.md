# RT59-v3 独立审阅与下一步 Codex 交接

日期：2026-09-20。本文是独立审阅和实施提案；未修改远端仓库、未提交 HPC 作业。

## 1. 实际读取的版本与证据边界

```text
repository: rular099/team_pytorch
active result branch: rt59-dual-objective-transport-v3
reviewed HEAD: d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd
result evidence commit: b709825b78e7579dbf05c8d4eacda5b29b9e58ae
previous handoff base: 40830acd475bd40347fef03427b52c4a04432dc3
actual implementation commit in compare history:
  54fd63d623ac5837e16a5d175d34041b7488661e
known training-counter fix:
  7e00824b3aab7a15286dfd9bf9a264b03080c1cd
```

原 RT58 分支仍停留在 40830acd；新工作确实位于 RT59 分支。先后读取了 AGENTS、PROJECT_CONTEXT、协作 README、SESSION_SUMMARY 的相关部分，以及当前结果、配置、模型、损失、训练循环、评估、launcher、测试和 commit compare。数据生成器未列入本轮变更；其 causal cutoff/波形 mask/标签隔离链路沿用此前核验路径。

原始 NPZ、实际 checkpoint 和完整日志没有挂载到本轮运行环境。本轮可认证的是代码语义、提交内容、可见报告及其算术一致性；报告中的真实 NPZ 重现和 5000 次 bootstrap 是 Codex 已提交的证据，不冒称本轮重新执行。

额外独立执行了 9 项算术/合成数据检查，见同附的 `review_math_checks.py/json`。其中将从连接器读取的 group_metrics.csv 文本重建为字节，并验证其 Git blob SHA 为 `b4554b8ce468a7bb1a0f9af1c5f8d095548397a9`。这些不是仓库测试或 HPC 性能试验。

## 2. 结论

RT59-v3 是有实质收益的双协议点预测候选，应保存作为下一轮 parent/reference，而不是丢弃或无变化续训。但整体仍不能标记完成：random slope、single-station pairwise MAE 和两个历史 coverage retention 条件仍失败；三个 range 条件需要修正统计定义后重算。

不要原样认证“严格按原门槛 22/29”。新 analyzer 把 P95−P05 range 改成了 max−min；还把原 handoff 中四个联合诊断提升为了 required gate。应保留原报告作为历史证据，另生成有版本标识的更正报告。

### 2.1 已提交的可靠点指标

固定 RT59 epoch8、seed42、validation，点估计为 log10(m/s²) 坐标中的 MDN mixture mean。base 是同次 NPZ 内冻结的 RT57 gamma=1.66，而不是 gamma=1。

| Population | Events contributing | Targets | Base MAE | RT59 MAE | Base RMSE | RT59 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| random non-input | 1,310 | 75,654 | 0.245578 | 0.237828 | 0.318243 | 0.311250 |
| normal all | 1,383 | 89,770 | 0.131016 | 0.125006 | 0.195544 | 0.186050 |
| normal input | 1,383 | 63,651 | 0.098452 | 0.094520 | 0.155096 | 0.148322 |
| normal non-input | 1,290 | 26,119 | 0.210371 | 0.199301 | 0.269814 | 0.255652 |

normal protocol 全部有 1,383 events；non-input 指标实际由其中 1,290 events 贡献，不能把这两个数字混用。两个 protocol 都有 9,681 realtime rows，但 random 只有 1,310 events 贡献有效 non-input targets。

MAE 相对下降：random 3.16%，normal all 4.59%，normal input 3.99%，normal non-input 5.26%。报告的上述 MAE/RMSE 配对 event-cluster CI 全部位于改善方向。

normal all 的 |error|≤0.2 dex 比例 75.64%→77.18%，P95 absolute error 0.42318→0.39873 dex；normal non-input 对应为 56.41%→58.72%、0.52744→0.49551 dex。说明误差点云确实有所收拢，但不是“大部分散点已经贴紧对角线”。

利用 MSE=bias²+Var(error) 做独立算术拆分：random 的 MSE 减少中约25.89%对应整体 bias² 减少，normal all 约31.44%；两者中心化误差方差也下降。因此收益不只是把所有点整体平移。此为代数分解，不是干预归因，不据此拟合新的 validation 校正器。

random triggered non-input：33,601 targets / 1,301 contributing events，MAE 0.230893→0.224602。untriggered：42,053 targets / 1,292 events，MAE 0.257311→0.248396。requested 1s：16,348 targets，MAE 0.255091→0.243391，明确通过0.253门槛。

normal 历史 RT55 epoch32 all-target MAE约0.1166仍更低。RT59的双协议收益是相对当前冻结RT57 base；不应宣称已经超过RT55所有normal指标。历史比较不是本轮独立完成的NPZ逐目标配对。

### 2.2 概率指标

random NLL 0.217530→0.199018，Brier 0.189875→0.183952；normal all NLL −0.822244→−0.848157，Brier 0.100965→0.095412。Brier阈值为−1.2 log10(m/s²)。

mean-only shift 保持每个样本的 mixture logits、component sigmas 和 mixture variance；coverage1变化 random +0.020713、normal +0.014860，超过原0.01工程retention预算。因此历史gate仍失败，但不能说这是“模型扩大sigma掩盖误差”或“NLL/Brier都恶化”。一般MDN的mean±sigma也不是精确的68.27%分位区间。

### 2.3 明确评估缺陷

1. `tools/analyze_rt59_dual_objective_npz.py::_field_metrics` 约L159–200使用`np.ptp`，而沿用的RT58阈值基于逐field P95−P05，至少5个valid targets。三个range gate及其delta需要更正。不能从现有极差汇总量反推正确分位数指标。
2. 原V3 handoff第8.3节将normal all/non-input的bias/slope作为联合诊断；新脚本4行`required=true`应单独标明。按25项必要条件加4项诊断拆开；不要改历史阈值或删除失败项。
3. `RESULT_REVIEW.md`把validation写成Japan 2018，与继承配置、全量报告、样本计数不一致。以实际resolved eval config与split/event manifest核对并纠正，不仅凭目录名断言年份。
4. 同一报告的implementation完整SHA写成`54fd63de2d17b771049943013d45265f32520137`；compare history实际是`54fd63d623ac5837e16a5d175d34041b7488661e`。需要更正并区分implementation、HPC source manifest、result evidence、review HEAD。
5. `_event_ids`在缺少event_id时回退到row id。当前报告显示实际用了event clusters；不能据此说本次CI有错，但修订分析器应fail closed，避免未来误把realtime rows当独立事件。
6. fixed-context roll目前汇总了“delta发生变化”，没有提交对应误差penalty。仅非零变化证明敏感性，不证明正确配对产生净收益。现有NPZ可直接补出差值，无需任何新forward或HPC control作业。

在现有29行表中，可保留22个PASS方向，4个与range无关的FAIL，3个range条件暂未按原定义确立。另将4个diagnostic从required计数分离后，是18个已通过必要项、4个确定失败必要项、3个待正确计算必要项。NO_GO不因此翻转。

## 3. 根因判断：限制结论的强度

已证实：输入/标签路由分开、local和transport参数分开、base缓存复用、所有MDN component同量平移。新损失有R/NI/NO全局DDP分母；训练循环用新objective替换旧汇总loss而非重复叠加。训练有8轮记录，两支路均有非零梯度。

有证据支持的推断：目前更紧迫的是transport输出/目标如何表达空间差分，而不是证明encoder会不会使用波形。actual-one-station时w=1，local分支不参与remote query；space failure不能仅归因于多台站aggregation。现有pair MLP明确接收query/geometry，不能直接套用“single-key attention使全部query必定相同”的解释。

一个需要承认的旧设计局限：relative auxiliary来自我上一轮方案，不是Codex擅自写错。

```text
E_j = y_j-b_j ; E_i = y_i-b_i ; e_i = a_i-b_i
relative_target_ji = E_j-E_i
candidate_ji = b_j + k_i*e_i + r_ji

若a_i=y_i且r_ji=relative_target_ji：
candidate_ji-y_j = (k_i-1)*E_i
```

```latex
E_j=y_j-b_j,\quad E_i=y_i-b_i,\quad e_i=a_i-b_i,
\quad r^*_{ji}=E_j-E_i,
\quad c_{ji}=b_j+k_ie_i+r_{ji},
\quad a_i=y_i,\ r_{ji}=r^*_{ji}
\Rightarrow c_{ji}-y_j=(k_i-1)E_i.
```

因此relative损失和最终candidate目标一般不是同一参数化的相容零误差目标，除非k接近1或输入base误差接近0。不过单站时这个多余项是field-common，做query差分会消掉；不能把它单独认定为range收缩的已证实原因。

另一个推断是：Huber差分目标的尾部梯度饱和与有限信息下的条件均值收缩可能使大空间反差欠拟合。不能把预测range达到真实range当作无条件正确目标：信息不足时条件均值本就可能比真实随机场更平滑。下一实验必须同时减少真实差分误差并保持点/概率精度，不能只拉range或斜率。

## 4. 唯一新实验：RT60，固定其余模型的空间残差读出再拟合

不是重新训练local，不是解冻DiTing，不是再叠加一套大head。

从真实RT59-v3 epoch8 checkpoint做weight-only初始化，**只训练**：

```text
pga_anchor_residual_transport_head.transport.residual_head.*
```

当前hidden=256时为6个parameter tensors、67,077个scalar parameters。冻结local、set/pair encoder、score、level、anchor、RT57及DiTing，冻结模块持续eval。这样station weights、level correction与输入站local结果都不会因参数更新而改变。

这个实验直接检验：现有冻结表示是否足以通过更一致、针对最终空间差分的读出优化，挽回空间反差，而不破坏已获得的双协议收益。它不能证明未来无需改进表示。

### 4.1 同次计算冻结RT59 parent，而不是错用RT57作新主对照

fresh transfer时复制一份`residual_head`为只读reference；不是复制整个模型，不是再次跑DiTing。snapshot存为checkpoint metadata中的独立`rt60_reference` payload，不注册成FullModel子模块，因此不改变旧model_state_dict的参数键。

payload需包括readout_state_dict、SHA-256、parent checkpoint SHA、父模型所有非readout张量fingerprint、parent epoch/task_id、normalization、source identity。resume必须从payload恢复原reference，禁止复制当前学生头成为reference。production推理不依赖外部reference文件；评估reference只读该checkpoint metadata即可。

同次forward保留detached `residual_input`、distance gate、weights；reference head在这些相同输入上执行一次轻量计算：

```text
r_old_ji = distance_gate_ji * ReferenceResidualHead(residual_input_ji)
r_new_ji = distance_gate_ji * StudentResidualHead(residual_input_ji)
change_j = (1-observed_j) * sum_i w_ji*(r_new_ji-r_old_ji)
mu_RT60_j = mu_RT59_reference_j + change_j
```

```latex
\Delta_j=(1-I_j)\sum_i w_{ji}(r^{new}_{ji}-r^{old}_{ji}),
\qquad m^{60}_j=m^{59}_j+\Delta_j.
```

旧`val_rt59_base_mean/mdn`仍然表示RT57 gamma=1.66，不得改名换义。新增独立`val_rt60_reference_*`表示RT59 parent。同时保留student final，形成可对齐的RT57/RT59/RT60三组预测。

### 4.2 损失：最终输出是真正监督对象

主损失保留NLL+0.10 SmoothL1(beta=1)+0.10 MSE，使用原R/NI/NO权重0.50/0.25/0.25和全局target分母。NI输出冻结，相关loss为常量，只作记账，不把其预算重分配给其他组。

RT60不再叠加旧relative/candidate auxiliary：二者weight=0；旧RT59 aggregator不能再执行。关闭旧Huber within-event auxiliary，以以下平方差分目标替换。

对于同一个realtime field内n>=2个合法remote queries，e_j=m_j-y_j：

```text
S_field = mean_(j<k) [0.5 * (e_j-e_k)^2]
        = sum_j (e_j-mean(e))^2 / (n-1)
```

```latex
S_f=\frac{1}{\binom n2}\sum_{j<k}\frac12(e_j-e_k)^2
   =\frac{1}{n-1}\sum_j(e_j-\bar e)^2.
```

它对field-common平移不敏感，输出梯度和为0，直接优化真实query差分，而不是range倍率。中心化**只在loss中**，禁止在forward里按当前query集合中心化，避免预测随请求集合变化。

contrast采用四个field buckets的全局均值：R单站、R多站、NO单站、NO多站。单站依据actual valid selected inputs，不用requested count。权重分别0.25/0.25/0.125/0.125，外乘0.40；空bucket为图连接零，不把权重分给其他bucket。相当于保持R/NO预算0.50/0.25，每组内单站与多站各一半，明确把优化预算给单站反差，而不更改sampler。

每field先算完整合法query pairs均值，再在bucket内对field等权。使用上述O(Q)等价式，不截取前105 pairs；在原Q<=15时正好覆盖原允许的所有pairs。NI不参与contrast，不建立NI−NO pair。

remote上的truth-regret权重0.05，比较**RT59 parent**与student相对同一真值的SmoothL1，变好不罚，变差才罚；此次同时应用R/NO。它是软保护，不是推理oracle，不保证validation逐target不退化。

所有输出与loss在既有train-normalized PGA坐标内，原mean/std保持不变；报告仍为raw log10(m/s²)。contrast除以std²的效果由先归一化目标自然产生，不再重复缩放。

### 4.3 优化与执行

一个新seed42实验，固定8个新epoch。Adam betas=(0.9,0.999)，eps=1e-8，weight_decay=0；lr为epoch1–4 1e-4，5–6 5e-5，7–8 2.5e-5。只对readout做norm1.0 clipping。新optimizer、scheduler、epoch计数，不继承旧optimizer。不得看到validation后续训或改lr/weight。

新任务仍用原80%random/20%normal训练、原固定validation、原train事件/normalization、原mask/cutoff、台站count集合、query exclusion。只在当前修改后代码中训练一个candidate，不做sweep或第二seed。

选择8轮和上述权重是前瞻固定工程设置，不是声称已经验证最佳。若readout无法从冻结特征恢复反差，结果将反证这条低成本路线的充分性；不自动扩大可训练范围。

### 4.4 评价分两层，不能改历史结果

- `rt60_mechanism_pass`：相对同次RT59 reference，actual-one-station pairwise-delta MAE与P95−P05 range absolute error的paired event-cluster95%CI上界均<0；random整体range absolute error点估计不恶化；random与normal non-input的MAE/RMSE点估计均不恶化；同址input预测逐target保持一致；random与normal all/non-input的NLL、Brier点估计不恶化。全部同时满足。
- `legacy_full_go`：仍按纠正后的原必要门槛计算，relative historical比较仍用RT57 gamma=1.66；不把RT59 reference偷偷替换成原表的base。四项bias/slope仍单列diagnostic。包括原coverage预算，不追溯改为pass。

mechanism_pass只代表研究路线有效，不代表已经达到legacy_full_go，更不代表held-out test或生产可用。如果研究条件未满足，保留RT59作为默认development parent，保存RT60负结果，不自动追加实验或选择中间epoch。

## 5. 可直接发送 Codex 的完整 prompt

```text
[AI-HANDOFF]
task_id: 20260920-rt60-final-contrast-readout
repo: rular099/team_pytorch
branch: rt59-dual-objective-transport-v3
base_commit: d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd
rt59_evidence_commit: b709825b78e7579dbf05c8d4eacda5b29b9e58ae
status: proposed_not_implemented

goal:
  修正现有RT59评估口径，并实施且只实施一个RT60实验：
  冻结RT59除空间residual_head之外的全部模型，用最终场平方差分目标
  改善one-station空间预测，同时保留RT59的random/normal点与概率收益。
  本prompt与附件第4节共同构成完整规格，不沿用早期RT59-v1/v2指示。

verified_facts:
  - 当前结果位于RT59新分支，不是旧RT58分支。
  - RT59 epoch8有random/normal双提升，NLL/Brier同向改善；保留该checkpoint。
  - analyzer::_field_metrics使用np.ptp，偏离历史P95−P05定义。
  - slope=0.402453<0.45；单站pair MAE=0.336202>0.330；
    两个coverage1历史budget仍失败。range原判定需修复。
  - 旧relative target与candidate在k不为1时一般不相容；
    这是旧方案局限，不能当成已证实的唯一空间收缩根因。
  - 当前residual_head为6张量/67077参数；仅改变这些参数即可保持local输出。

inferences:
  - 当前优先研究输出端的空间差分学习，而不是重新证明波形有效或先解冻DiTing。
  - 仅readout再拟合是否足够必须由此一实验检验；不预先保证成功。

files_to_inspect:
  - AGENTS.md; docs/ai/PROJECT_CONTEXT.md; docs/ai/README.md; SESSION_SUMMARY.md
  - docs/ai/CODEX_RESULT_20260920_RT59_DUAL_OBJECTIVE_TRANSPORT_V3_VALIDATION.md
  - reports/rt59_dual_objective_transport_v3_validation_20260920/*
  - docs/rt59_dual_objective_transport_v3.md
  - gemini_models.py; train_light.py; gemini_util_light.py; eval_checkpoint.py
  - tools/rt59_dual_objective.py; tools/analyze_rt59_dual_objective_npz.py
  - tools/run_rt59_dual_objective_transport_v3_slurm.sh
  - tests/test_rt59_dual_objective_transport.py 和RT55–RT58已有兼容性测试
  - 附件RT59_V3_IMPLEMENTATION_AND_AI_HANDOFF_20260915.md第8.3节

constraints:
  - 先核对真实HEAD/worktree；HEAD不一致先审查差异，不强制reset或覆盖用户文件。
  - 不修改旧RT55–RT59 config字节、模型默认值或model_state_dict键/shape。
  - 原关闭开关加载/推理/导出schema完全保持；新行为用新config及显式训练/评估开关。
  - 不使用held-out test；不重跑RT59训练/validation/smoke/query diagnostics。
  - 不做超参数、station-count、seed或checkpoint sweep。
  - 不解冻encoder、TEAM、anchor、local、pair/set/score/level。
  - 不把protocol bit、input/query PGA标签或未来触发状态送入forward。
  - 不按当前query集合做forward中心化，不给输入数组顺序任何新的语义。
  - 不强行scale预测range/slope，不为通过coverage而改sigma。

phase_A_existing_artifact_correction:
  - 只读现有两份RT59 NPZ与metrics/config/log包，不做任何新模型forward。
  - tools/analyze_rt59_dual_objective_npz.py：
    恢复逐field P95−P05，至少5个valid targets，逐field等权；
    quantile interpolation和近零truth range处理与历史canonical helper相同。
    max−min如保留，只能加独立ptp_*诊断，不能同键换义。
  - 重新生成三个range gates；保留阈值不动。
    明确分开25个required与4个diagnostic，保留原29行及旧输出可追溯。
    如果发现另有事前签署的更强gate规范，展示其commit/文本，不能静默决定。
  - event_id缺失/实际站数与mask冲突/非法单位/基线不对齐时fail closed。
    核对formal input和observable route逐元素，而不是仅对比总计数。
  - 修正Japan2018文本与implementation SHA；从真实resolved配置和event manifest
    提交year/split/count摘要，脱敏路径但保存原始文件SHA与脱敏文件独立SHA。
  - 已有fixed_context_rolled_delta补算correct-vs-roll MAE/RMSE差，
    single station必须恒等；多站单列。仅修订NPZ分析器，不新增roll job。
  - 同次导出field-level轻量统计表：event_id、time、actual_station_count、n_targets、
    base/final P95−P05、ratio、range_abs_error、pair MAE/RMSE、
    field-mean error和centered error。它使下一轮ChatGPT可独立复核，不需360MB NPZ。
  - 新目录reports/rt59_v3_review_correction_20260920；不覆盖旧报告。
    旧报告只加更正入口，原数据/29行保持归档。
  - 如果真实NPZ/schema/身份不满足协议，停止训练并报告具体阻断；
    仅旧文档笔误或本次已定位的ptp bug不要求重跑HPC。

phase_B_single_experiment:
  - 新config继承RT59-v3；标识RT60 final-contrast-readout；固定8新epoch seed42。
    使用真实RT59 epoch8作weight-only parent，核验task_id、epoch、checkpoint SHA。
    不执行initialize_rt59_from_rt58，不做warm-copy，不恢复RT59 optimizer。
  - 唯一trainable prefix：
      pga_anchor_residual_transport_head.transport.residual_head.
    实测assert6 tensors/67077 scalars（当前production配置）；任何差异先报告。
  - 新freeze_mode=rt60_contrast_readout_only。
    model整体eval，再只将该readout设train；冻结buffers不更新。
  - 实现第4.1节轻量reference头snapshot；保存在checkpoint metadata中，
    不注册为FullModel模块；resume恢复最初RT59 reference，不重设为student。
    同次缓存h/distance/w，reference只执行其readout；不重跑encoder或完整base。
  - 实现第4.2节全部loss/reductions：
      point=NLL+.10 SmoothL1(beta1)+.10 MSE，R/NI/NO=.50/.25/.25；
      regret=.05，所有remote targets，相对RT59 reference；
      relative_weight=0，candidate_weight=0；
      contrast=.40*[.25 R_single + .25 R_multi + .125 NO_single + .125 NO_multi]；
      每bucket是field平均，每field是完整pairs的half squared difference平均；
      NI禁止加入contrast；空bucket图连接零、全局计数，不重新分配预算。
    旧RT59 objective和旧PGA auxiliaries不得重复计入。
  - 标准DDP梯度平均下，每rank本地sum乘world_size/global_count。
    所有rank执行相同collective顺序，含本地空组；target/field分母分开。
  - Adam lr1e-4(epoch1–4),5e-5(5–6),2.5e-5(7–8)，
    betas(.9,.999),eps1e-8,wd0，readout独立clip_norm1。
  - 原sampler80/20、station counts、cutoff、normalization、query targets不改。
  - 同一个epoch8 checkpoint做random/normal validation，不选中间epoch。

proposed_change:
  - tools/analyze_rt59_dual_objective_npz.py及新统计测试：phase_A口径修复；
    不影响旧网络输出。
  - 新tools/rt60_contrast_objective.py：group/field loss、reference快照协议。
  - train_light.py：新freeze/optimizer/scheduler/metadata resume分支；
    新objective替换分支，完整日志和精确trainability验证。
  - gemini_models.py：default-off缓存暴露、reference轻量解码接口；
    不改旧readout层形状/层数/参数键，不改变旧RT59输出路径。
  - eval_checkpoint.py：conditional val_rt60_reference_mdn/mean、
    val_rt60_increment、reference identity导出；
    旧val_rt59_base_*仍指RT57，不换含义。
  - 新pga_configs/transformer_japan_full_2000_2024_rt60_contrast_readout_seed42_chaosuan.json
    和normal_validation companion；旧config完全不动。
  - 新tools/run_rt60_contrast_readout_slurm.sh、tests/test_rt60_contrast_readout.py、
    docs/rt60_contrast_readout.md和对应CODEX_RESULT。

acceptance_checks:
  - range测试覆盖ptp与P95−P05会导致不同gate的合成反例，
    不只测试JSON能写出或gate行数。
  - 修订分析器的旧point/probability统计与原已提交值在既有容差内一致。
  - step0 candidate等于RT59 reference；保存/恢复后reference不随student漂移。
  - 一次optimizer.step后仅6个readout参数可变，
    所有非readout参数/buffers fingerprint保持一致，station weights/level不变。
  - observed local输出逐target不变；不只是local参数requires_grad=false。
  - final60=reference59+new_increment；全部MDN logits/sigmas不变。
  - pair half-MSE与中心化error式一致；field-common平移不变，输出梯度和为零；
    无效NaN在算术前mask；single/empty/multigroup均覆盖。
  - 两rank不均衡/空组DDP与拼接样本全局目标梯度一致；
    可本地CPU测试，不新增HPC smoke。
  - query/station联合重排和query分块不变；不增加encoder调用次数。
  - 一次完整tiny训练+validation+checkpoint/resume测试，覆盖实际train_model分支。
  - 新局部测试及RT55–RT59加载/推理回归；py_compile、bash -n、git diff --check。
  - 不把本地测试/代数测试称为实际多节点/HPC性能结果。

metrics_and_decision:
  - primary comparison=RT60 minus same-forward frozen RT59 reference。
    historical comparison=RT60 versus same-forward frozen RT57 gamma1.66，分栏输出。
  - 报告all/input/non-input/triggered/untriggered、时间与actual station count；
    每指标给target/field/event计数，PGA raw log10(m/s²)，mean estimate不换。
  - 5000次paired event-cluster bootstrap seed20260915，
    field指标保留逐field等权估计量，同一事件的重复时刻一起重采样。
  - 预期random 9681rows/1310 contributing events/75654targets；
    normal 9681rows/1383events/89770targets，input63651/noninput26119；
    random>=5target fields5762，其中actual-one-station1494。
    成功前不能通过删目标使计数改变。
  - 第4.4节rt60_mechanism_pass与legacy_full_go分开；
    不因部分改善更改NO_GO，不把更正统计算作新训练收益。
  - 统一尺度图和P90/P95绝对误差、固定0.1/0.2dex比例；不删离群点。
    同时NLL/Brier/predictive sigma/coverage；coverage历史规则照报。

hpc_followup:
  - 仅一个train -> random+normal两个afterok验证；无test、额外roll、smoke或sweep。
  - launcher默认DRY_RUN=1、ACTION=train|eval|all、显式CONFIRM_RT60。
  - 实际运行沿用用户既有HPC提交授权；没有提交权限/授权时只准备dry-run交接，
    不伪造完成状态、不索取私钥。
  - 新空输出目录；env使用WORKDIR/JAPAN_FULL_DATA_ROOT/RT59_PARENT_CHECKPOINT/
    RT59_PARENT_CHECKPOINT_SHA256/RT60_WEIGHT_PATH，不硬编码新增私有路径。
  - exact clean Git SHA或完整uploaded source manifest；包含所有transitive配置依赖，
    特别不能漏掉RT55继续extends的更早配置。
  - 保存parent/encoder/config/source/split身份、new task_id、每epoch各组目标数与loss、
    optimizer/lr/gradient、teacher fingerprint、job ids、fixed epoch8 metrics/NPZ。
  - Github提交轻量更正报告/新结果/逐field统计/概率汇总/可执行分析脚本；
    checkpoint和大NPZ留HPC，不要求上传大文件到Git。

risks:
  - 旧表示或固定pair权重不够时，仅readout再拟合可能无效；不自动放开冻结范围。
  - squared contrast重视大差分，可能追逐噪声、损伤MAE/NLL；严格parent对照防止误判。
  - conditional mean范围比真实field小不必然是错误，range放大必须伴随误差改善。
  - reference快照/resume和旧base语义容易混淆，需要同次重建测试。
  - 单seed/development重复使用，CI不覆盖训练seed或方案选择不确定性。

open_questions:
  - none requiring a new scientific choice；
    真实checkpoint路径与SHA从既有HPC产物解析，不猜测。

required_return:
  - 先说明phase_A实际更正结论与canonical gate状态，再报告实现/测试/HPC各阶段。
  - 按docs/ai/README.md返回CODEX-RESULT，含task_id/base/result_commit/branch/
    changed_files/verification/compatibility/hpc_status/remaining_risks/review_request。
  - 若已有预期之外的新commit或job，先报告真实状态，不覆盖，不机械套本prompt。
[/AI-HANDOFF]
```

## 6. 本轮依据的源文件

以下均锁定HEAD `d43c7d63528b3ade71c3506eeb2cb71f90fb3ffd`：

- `reports/rt59_dual_objective_transport_v3_validation_20260920/RESULT_REVIEW.md`：protocol、汇总、训练、局限。
- 同目录`summary.json`（含完整strata、tail errors和CI）、`group_metrics.csv`、`training_summary.csv`、`forward_control.json`、`gates.csv`、`artifact_manifest.sha256`。
- `tools/analyze_rt59_dual_objective_npz.py` L45–202：event fallback、point/CI/field算法。
- `gemini_models.py` L1659–1950：local、transport、route；L5420–5610：cached decoder；L6300–6465：RT59/RT58互斥集成和roll。
- `tools/rt59_dual_objective.py` L1–506：group/DDP/relative/contrast真实损失。
- `train_light.py` L2470–2660：真实训练循环里objective替换与统计。
- `eval_checkpoint.py` L1280–1435：公共标签/预测导出、NLL normalization Jacobian。
- RT59 config、launcher、tests：执行与兼容性约束。
- 上一轮用户可见的`RT59_V3_IMPLEMENTATION_AND_AI_HANDOFF_20260915.md`第8.3节，以及RT58报告第5节：历史gate/分位数定义。

本文没有把报告中尚缺的checkpoint-body SHA、canonical quantile数值或rolled prediction误差补造成“已知结果”。
