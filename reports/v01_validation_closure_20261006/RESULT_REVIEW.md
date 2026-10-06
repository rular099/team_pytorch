# V01 validation closure：结果汇总与审阅材料

日期：2026-10-06（Asia/Shanghai）。仓库 `rular099/team_pytorch`，分支
`exp/v01-velocity-prep-padding-control`，分析基线
`7d87c4007b78405bb901c67ff0b78dc5d0e85a8e`。
原始回传：`../chaosuan_res/vel/validation_closure_v1_retry1`（相对仓库根），
四格历史速度 random 原文件：`../chaosuan_res/vel/eval_retry1`。原产物保留不改。

## 1. 结论先行

**工程验证闭环已完成；不等于速度优于加速度、补零瓶颈已被解释，或论文级 V01 全部完成。**
真实 idx7 根因确认，五格 normal 与一格 AA random 成功导出；四格速度 random 复用并认证
真实 sensor ID/UTC/标签及有界输入签名。三臂仍用原 epoch8 / optimizer step1496 权重，
没有新训练、preflight、held-out test、模型修改或阈值调整。

科学结果延续首轮反预期现象：

- 缺失训练臂的总体 MAE/RMSE/NLL/Brier 优于完整训练臂，在 normal/random 均成立。
- 主要差异与训练臂相联系；固定模型仅切换缺失视图，效应小，尤其 M 模型近零。
- 总体改善主要来自 bias² 减少，不能称误差云明显收紧或已证明正则化。
- PGA 阈值以上目标的 MAE/NLL/Brier 在两种场景均退化，早期 1/3/5s 也一样。
- AA 比 FF 好，但和 MM 的共同可预测群体 MAE 差异 CI 跨零；不能宣布速度或加速度胜出。
- 单输入台站空间场仍严重压缩，不能用单点 MAE 改善宣称空间辨识改善。

本轮仅整理与核验；**不自动追加超算任务**。请 ChatGPT 审阅后决定是否按限定的负机制
结果收尾，或说明一个确实影响结论、且不能靠现有材料回答的最小缺口。

## 2. 实验与指标口径

F = 完整速度前缀；M = 同一速度记录删除模板指定的部分 P 前有效样本，填零且 mask=False；
A = 与 Hi-net source 配对的实测加速度输入。FF/FM/MF/MM 的首字母是训练臂，次字母是推理视图。

固定 seed42、8 个新 epoch、step1496，复用 RT55 ep32 weight-only 起点和 RT55 模型。
normal 是 V01 原可用 source 选择路径；random 是截至同一决策时刻的 causal-random source
选择路径。两者均预测原 KNET query-only 目标；不是 PGV。

**V01 normal 的 all 是全部有效 query-only 目标，不是历史 RT55 含容易输入台站目标的
normal-all。** Hi-net/配对 A source 与 KNET query 分开；全部 sampler 目标为非输入类型，
sensor pairing 仍可能涉及同一台址，所以真正物理 site-remote 未认证。
因此不能直接把本表和 RT55 normal-all、RT59/RT61 或文献 test 排名比较。

PGA/error 坐标 `log10(m/s²)`，点预测为该坐标内 MDN mixture mean；NLL 也在这个坐标，
不要取 `nll_model_space` 混比。Brier 阈值固定 -1.2；above/below 指目标 PGA，不是事件震级。
coverage 是 mixture mean ± k·mixture std 的覆盖率，不是另算的等尾概率区间。

all7 = 1/3/5/10/20/40/90s；early135 = 1/3/5s 合并，不是三个独立 event 样本。
配对键：event ID + query sensor ID + requested time + exact absolute UTC cutoff。
CI：按 event 聚类配对重采样，5,000 draws，seed20260915，同事件全部时间/目标一起抽样。
target-micro 按目标加权；event-macro 按事件等权；macro RMSE 为各 event RMSE 的均值。
同一 seed 的 CI 不包含训练 seed 不确定性；分层属探索性，未做多重比较校正。

## 3. 本轮工程闭环证据

### 3.1 真实 idx7 根因

三份 trace：FF normal、FM normal、AA normal。真实事件 `20041029141300`，
第一个 shard base idx7 / local request49，1s；source 为 `N.KMOH`（AA 为 `NIGH06`），
query 为 `NIG019`，query P sample=-109，但实测 PGA 与查询坐标合法。
旧 clock 将 query validity 原地改成 False，实际复现 `Found event without PGA idx=7`。
修复后 changed_slots=0，query_count=1 / input_count=1，相同事件、相同 cutoff，真实样本恢复。
两版时钟均 current_sample=600、cutout=601、first_p_pick_sample=500；绝对 UTC
`1099026844.7340753` 不变。见 `evidence/audit/*.idx7_trace.json`。

这是本轮真实失败点证据，不再只是合成别名反例。仍不声称消除了继承的在线因果性风险。

### 3.2 权重与代码身份

- 三臂 before/after checkpoint 清单逐字段相等；epoch8、step_min=step_max=1496。
- 三臂 init SHA 相同；last SHA 与预注册合同相同，详见前后清单，不根据文件名认身份。
- 本地核验回传的 runtime manifest 全180文件与分析基线一致，6份 prepared config 哈希一致。
- source manifest SHA256：`88d4570c2f87c3f0daf00eece6e87dc0a733c78a4d86963645152d1538d9fc86`。
- 回传原 split SHA 与 preflight protocol lock 相符，train/dev 互斥，全部请求事件属于 frozen dev。
- 当前 encoder / parent / YAML SHA 已有；**原训练时 encoder/source 哈希与逐批采样轨迹仍缺失**。
  不能用这次重新计算的当前 hash 伪装历史训练证据；训练 scalar manifest 不是源码 manifest。

本地没有重读大 checkpoint body 或 HDF5；真实源行来自 HPC trace，权重 hash 来自 HPC
前后审计。本地直接读取所有10个 NPZ/身份 ledger，复算指标与主要 CI，见 `verification.json`。

### 3.3 完成状态和请求分母

| 场景 / arm | requested event-time | predicted | no source | 可预测 events | 选中 targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal，四格速度各自 | 8,358 | 8,358 | 0 | 1,194 | 84,259 |
| random，四格速度各自（复用） | 8,358 | 8,358 | 0 | 1,194 | 84,259 |
| normal，AA | 8,358 | 8,324 | 34 | 1,193 | 83,995 |
| random，AA | 8,358 | 8,313 | 45 | 1,193 | 83,934 |

AA normal/random 请求弃权率 0.4068% / 0.5384%；invalid_label/implementation_error 均0。
normal 的34请求涉及10个事件，random 的45请求涉及15个事件，不能混用请求数和事件数。
全时刻无预测事件为 `20240728191300`。无预测请求完整保留于 `no_prediction_requests.json`。
候选 query 机会（sampling前）138,082；弃权请求内候选机会 normal538/random601，
**不能把它们与选中 targets 相除冒称 sampling 后目标弃权率**。

旧 AA random 的15组重复、45额外行已追踪；新版8,313个预测 event/time 唯一，没有邻居替代。
上述45是请求数，不是45个独立事件。旧文件保留，不去重、不覆盖。

HPC job IDs 为29318960–29318967（audit、6eval、analysis）；应用层 ledger/report/guard 完整。
**sacct 未回传，不单凭产物给调度器写8项 COMPLETED，也不报告不存在的耗时/MaxRSS。**

## 4. 总体结果

下表均 all7，目标为各系统成功预测的全部有效 query-only；AA 和速度分母不同，
跨模态必须看第7节共同群体，不按本表直接比较胜负。

| geometry | cell | targets / events | MAE | RMSE | R² | slope | bias | NLL | Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | FF | 84,259 / 1,194 | 0.215428 | 0.276199 | 0.454788 | 0.529671 | 0.094123 | 0.042431 | 0.175739 |
| normal | FM | 84,259 / 1,194 | 0.214986 | 0.275815 | 0.456305 | 0.526166 | 0.091680 | 0.040642 | 0.175288 |
| normal | MF | 84,259 / 1,194 | 0.207137 | 0.268345 | 0.485357 | 0.529469 | 0.070366 | 0.002226 | 0.166787 |
| normal | MM | 84,259 / 1,194 | 0.207152 | 0.268386 | 0.485198 | 0.527551 | 0.069652 | 0.001935 | 0.166702 |
| normal | AA | 83,995 / 1,193 | 0.206212 | 0.268059 | 0.485068 | 0.514002 | 0.055077 | 0.001317 | 0.166209 |
| random | FF | 84,259 / 1,194 | 0.244516 | 0.310286 | 0.307216 | 0.399227 | 0.091564 | 0.171931 | 0.202340 |
| random | FM | 84,259 / 1,194 | 0.244091 | 0.309952 | 0.308705 | 0.395758 | 0.088778 | 0.170561 | 0.201869 |
| random | MF | 84,259 / 1,194 | 0.236893 | 0.303735 | 0.336161 | 0.399740 | 0.066403 | 0.142036 | 0.194258 |
| random | MM | 84,259 / 1,194 | 0.236895 | 0.303782 | 0.335954 | 0.397787 | 0.065769 | 0.141841 | 0.194188 |
| random | AA | 83,934 / 1,193 | 0.235016 | 0.302155 | 0.342238 | 0.383159 | 0.048389 | 0.135702 | 0.192189 |

early135：速度各格36,111 targets / 1,194 events；normal FF/MM MAE=0.245445/0.237290，
random FF/MM=0.259122/0.250406。AA early normal=0.236302（35,946/1,188），
random=0.249310（35,889/1,187）。详见 `headline_metrics.csv` 和原分层表。

normal FF/MM 的1σ coverage=0.628360/0.656203，2σ=0.943199/0.949418；
random FF/MM 的1σ=0.619934/0.640133，2σ=0.940897/0.943460。
MM 的概率输出也有改善，但不能只凭一个覆盖率判定完全校准。

## 5. 训练臂效应、推理视图效应和较大 PGA 退化

差值统一为后者减前者；误差指标负值为后者更好。all7 / 同1194 events / 84259 targets。

| 比较 | normal ΔMAE [95% CI] | random ΔMAE [95% CI] |
| --- | --- | --- |
| MM−FF（联合差异） | -0.008275 [-0.009000,-0.007552] | -0.007620 [-0.008285,-0.006954] |
| MF−FF（固定F视图换训练臂） | -0.008291 [-0.009024,-0.007582] | -0.007622 [-0.008278,-0.006972] |
| FM−FF（固定F模型删前缀） | -0.000442 [-0.000710,-0.000147] | -0.000425 [-0.000698,-0.000142] |
| MM−MF（固定M模型删前缀） | +0.000015 [-0.000185,+0.000231] | +0.000002 [-0.000187,+0.000199] |

F 模型的微小 view 效应在 target-micro CI 中可分辨，不能说完全无效；但它远小于换训练臂，
event-macro 的对应 CI 均跨零。M 模型 view 效应 CI 跨零。不能从单 seed 两个训练过程
推断已经证明正则化；也不能把统计可分辨的小差异夸成显著应用提升。

阈值上下的 MM−FF（all7）：

| geometry / stratum | targets / events | ΔMAE [95% CI] | ΔNLL [95% CI] | ΔBrier [95% CI] |
| --- | --- | --- | --- | --- |
| normal，PGA≥-1.2 | 35,917 / 1,051 | +0.005165 [0.004164,0.006177] | +0.025704 [0.020790,0.030661] | +0.015212 [0.014119,0.016385] |
| normal，PGA<-1.2 | 48,342 / 1,158 | -0.018261 [-0.019227,-0.017319] | -0.089681 [-0.095294,-0.084046] | -0.027054 [-0.028510,-0.025603] |
| random，PGA≥-1.2 | 35,612 / 1,051 | +0.009195 [0.008323,0.010055] | +0.041495 [0.037650,0.045329] | +0.019042 [0.018029,0.020109] |
| random，PGA<-1.2 | 48,647 / 1,158 | -0.019930 [-0.020778,-0.019105] | -0.082494 [-0.087061,-0.078064] | -0.028060 [-0.029295,-0.026883] |

early135 阈值以上 ΔMAE：normal +0.008667 [0.007243,0.010142]（15,483/1,051）；
random +0.009951 [0.008628,0.011306]（15,286/1,051）。方向不是只由晚时刻带来。
micro/macro MAE 的阈值以上方向一致，细节见完整 CI；不根据这些结果改阈值、挑epoch。

## 6. MSE 与空间场

终端公式（同一目标群体，error=prediction−truth）：

```text
MSE = mean(error²) = bias² + mean((error − bias)²)
bias = mean(error)
```

原始 LaTeX：`\mathrm{MSE}=b^2+\frac1N\sum_i(e_i-b)^2,\quad b=\frac1N\sum_i e_i`。

| geometry | MSE FF→MM | bias² FF→MM | centered variance FF→MM | Δcentered variance [95% CI] |
| --- | --- | --- | --- | --- |
| normal | 0.076286→0.072031 | 0.008859→0.004851 | 0.067427→0.067180 | -0.000247 [-0.000583,+0.000089] |
| random | 0.096277→0.092284 | 0.008384→0.004326 | 0.087893→0.087958 | +0.000065 [-0.000243,+0.000362] |

两个场景的 centered variance 差异 CI 均跨零，总体 MSE 改善主要体现偏差改变。
较大 PGA 的负 bias 反而加深：normal -0.068886→-0.093335；random -0.117053→-0.142999。
没有给预测做 validation bias correction。

单输入、至少5个 query 的空间场（all7）：

| geometry | eligible fields / events | max-min range ratio median FF→MM | pairwise-difference MAE FF→MM |
| --- | --- | --- | --- |
| normal | 523 / 416 | 0.006455→0.006371 | 0.353222→0.353216 |
| random | 1,514 / 781 | 0.006364→0.006350 | 0.354286→0.354288 |

空间范围仍极小。P95−P05 ratio **mean** 另存在 `field_p95_p05_ratio_mean.csv`；
不能和 max-min ratio **median** 混成同一指标或把新口径当事后 gate。

## 7. A/V 共同可评估比较

不使用旧带重复 AA 数据。共同 query keys + exact UTC 一对一，labels一致。

| geometry / common population | targets / events | AA MAE | FF MAE | MM MAE | MM−AA [95% CI] |
| --- | --- | ---: | ---: | ---: | --- |
| normal，common available | 83,995 / 1,193 | 0.206212 | 0.214872 | 0.206574 | +0.000362 [-0.001159,+0.001832] |
| random，common available | 83,162 / 1,193 | 0.234640 | 0.243474 | 0.235796 | +0.001156 [-0.000174,+0.002464] |

FF−AA CI 分别为 +0.008660 [0.006994,0.010284]、+0.008834 [0.007337,0.010292]。
AA 对 FF 有优势，但对 MM 的 MAE 差异不足以支持明确胜负，CI 跨零也不证明等效。
normal 速度-only unmatched264；random AA-only772、速度-only1097，均已保留名单。

common-remote **paired-sensor exclusion proxy**：normal83,245/1,189，AA/FF/MM MAE
0.206296/0.215073/0.206784；random82,655/1,191，0.234821/0.243711/0.236032。
这是已选 source 的 paired-acc sensor ID 排除，未认证真实不同物理台址；该组这里只报告
描述指标，没有新增专门 remote CI。详见 `acc_velocity_common_population_metrics.csv`。

共同群体是“两个系统都能预测”的条件性比较；需与完整请求覆盖率一起看，不能掩盖 AA
弃权和选择偏差。A/V 不同传感器、深度、场地、频响和输入域，也不是 padding 单因素因果对照。

## 8. 数据规模与保留限制

preflight 最终物化：2004–2024共21年、train7967/dev1194 events、source rows120101、
query rows142597。loader eligible counts 相同；train selected source/query metadata rows226414，
dev36284。这些不是训练重采样后实际 unique exposure 或逐批轨迹。
初始 metadata cohort 为8439/1225，不能和最终物化数量混用。速度 archive 部分年度不完整，
不能称全量完整日本速度数据。原 test 波形未物化，test不参与本轮。

mask正确不等于编码器完全不受缺失影响；当前是“删除前缀+合法mask”联合干预，
不是全面无P前波形或P后丢失实验。真实干预剂量与 train-only template 分布见原10-03报告
`intervention_support_audit.json` 和本轮 protocol lock；不扩大剂量追求预设结论。

仍缺/仍限制：历史训练runtime/encoder tensor identity和完整实际采样计划、严格在线因果性
认证、raw归一化前post-P等同性的完整证据、单seed/短训练、test泛化、真正物理site-remote。
原 full-record centering 与 cutout+1 数值保持不变，不能把此次验证闭环包装成因果认证。

## 9. 材料与复算

原始数据实际位于仓库上一层 `../chaosuan_res/vel/validation_closure_v1_retry1`，
历史 random 位于 `../chaosuan_res/vel/eval_retry1`。原产物未修改。

```bash
python reports/v01_validation_closure_20261006/prepare_review.py \
  --closure-root ../chaosuan_res/vel/validation_closure_v1_retry1 \
  --old-eval-root ../chaosuan_res/vel/eval_retry1 \
  --output /tmp/v01_closure_review_new
```

工具仅读原NPZ/ledger/config/split并写新报告，无模型推理或缓存重建。包内保存：
完整250行分层metrics、完整分层CI、主要FF/MM event sufficient statistics、两种空间指标、
A/V common指标、coverage/outer match/弃权名单、checkpoint前后清单、三个真实idx7 trace、
source/config/protocol/cohort证据、原图PNG和bin-count源表、精确copy SHA清单。
原118MB全比较sufficient表、raw NPZ、完整ledger/plan、checkpoint、波形不入Git。

`pga_density.png` 是相同固定坐标/色标的FF/MM normal/random四面板；
`residual_density.png` 是阈值上下残差对比。已视检、复算bin计数；残差图legend局部靠近峰顶，
是原图复用，未作论文级重新排版，不影响指标。SVG原文件哈希可在源目录复核。

本轮审阅请求见 `docs/ai/CHATGPT_REVIEW_REQUEST_20261006_V01_CLOSURE.md`。
