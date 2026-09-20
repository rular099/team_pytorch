# RT60 final-contrast readout

RT60 是从唯一、精确核验的 RT59-v3 epoch-8 `full_model_last.pth` 出发的单一机制实验。
它不重新训练 encoder、station adapter、RT57/RT58 模块或 RT59 的 local/transport
表示，只更新 transport 分支最终 `residual_head` 的 6 个张量（67,077 scalars）。新行为
由 `use_rt60_contrast_readout=true` 和 `freeze_mode=rt60_contrast_readout_only` 显式启用；
旧 RT55–RT59 配置的参数键、shape、加载和推理语义保持不变。

## Forward 与只读 reference

加载 RT59 parent 后，程序在任何更新前复制最终 readout 的 6 个张量作为不可训练的
RT59 reference。reference 和 student 复用同一次 forward 已缓存的输入，不重跑 DiTing
encoder；只额外执行一次轻量 readout。checkpoint metadata 保存：

- reference tensor state 与精确 SHA-256；
- parent checkpoint SHA-256、epoch 和 task contract；
- parent checkpoint/已加载模型的 non-readout fingerprint；
- normalization、config 和 source identity。

resume 必须从 metadata 恢复最初的 RT59 reference，不能把已更新的 student 当作教师。
旧 `val_rt59_base_*` 继续表示 RT57 gamma=1.66 historical base；RT60 另行导出
`val_rt60_reference_mdn`、`val_rt60_reference_mean` 与 `val_rt60_increment`。

observed-input route 的 increment 恒为零。标签、未来信息和 random/normal 标志均不进入
forward 路由。

## Objective

终端形式（`e_j = prediction_j - truth_j`）：

```text
                    1       ┐
S_field = ----------------- Σ  0.5 * (e_j - e_k)^2
          n * (n - 1) / 2  j<k

          1       ┐             2
        = -----   Σ  (e_j - mean(e))       , n >= 2
          n - 1   j
```

原始 LaTeX：

```latex
S_{\mathrm{field}}
= \frac{1}{\binom{n}{2}}\sum_{j<k}\frac{1}{2}(e_j-e_k)^2
= \frac{1}{n-1}\sum_j(e_j-\bar e)^2,\qquad n\ge 2.
```

实现使用右侧的 O(Q) 等价式。contrast 只覆盖 remote R/NO query，不建立 NI 与 NO 的
交叉 pair；按 actual station count 划分 R-single、R-multi、NO-single、NO-multi，bucket
权重依次为 `0.25/0.25/0.125/0.125`，整体 contrast 权重为 `0.40`。每个 bucket 按
全局 field 均值归约；空 bucket 贡献零且不重分权重。

主 point loss 是 `NLL + 0.10 SmoothL1(beta=1) + 0.10 MSE`，R/NI/NO 权重仍为
`0.50/0.25/0.25`。remote truth-regret 权重为 `0.05`，只惩罚 student 相对同次 RT59
reference 的真值误差退化。旧 RT59 relative/candidate/difference auxiliaries 在 RT60
配置中全部关闭，避免重复计入。

## 固定实验协议

- seed 42；固定训练 8 个新 epoch，只评价最终 epoch 8。
- Adam：betas `(0.9, 0.999)`、eps `1e-8`、weight decay `0`。
- LR：epoch 1–4 为 `1e-4`，5–6 为 `5e-5`，7–8 为 `2.5e-5`。
- 只对 readout 做 norm-1 gradient clipping。
- 全量 Japan 2000–2024 training shards、sampler 80/20、station set、cutoff、split 和
  normalization 均继承 RT59，不运行 test、smoke、roll、sweep 或额外续训。
- 一次 train 后仅提交两个 `afterok` validation：deterministic random geometry 与 normal。

## 超算 dry-run 与提交

超算不能连接 GitHub、使用人工上传目录时，先在上传目录中执行 dry-run。parent 路径必须
显式指向已有 RT59 retry1 的 epoch-8 checkpoint；launcher 不会猜路径，并要求同目录存在
训练时写出的 `config.json`。

```bash
cd /public/home/test_bigmodel/seismogram/zb/team_pytorch/team_pytorch_query_geometry_diagnostics

export WORKDIR="$PWD"
export RT59_PARENT_CHECKPOINT="$WORKDIR/weights_japan_full_2000_2024_rt59_dual_objective_transport_v3_seed42_retry1/full_model_last.pth"
test -s "$RT59_PARENT_CHECKPOINT"
test -s "$(dirname "$RT59_PARENT_CHECKPOINT")/config.json"
export RT59_PARENT_CHECKPOINT_SHA256="$(sha256sum "$RT59_PARENT_CHECKPOINT" | awk '{print $1}')"
export RT60_WEIGHT_PATH="$WORKDIR/weights_japan_full_2000_2024_rt60_contrast_readout_seed42"

SOURCE_IDENTITY_MODE=uploaded_sha256 \
ACTION=all DRY_RUN=1 \
bash tools/run_rt60_contrast_readout_slurm.sh
```

记下 dry-run 输出的 `[INFO] source_manifest_sha256=...`。确认目标输出目录不存在或为空，
再将该值原样填入并提交：

```bash
export EXPECTED_SOURCE_MANIFEST_SHA256=<dry-run打印的64位source_manifest_sha256>

SOURCE_IDENTITY_MODE=uploaded_sha256 \
EXPECTED_SOURCE_MANIFEST_SHA256="$EXPECTED_SOURCE_MANIFEST_SHA256" \
ACTION=all DRY_RUN=0 CONFIRM_RT60=1 \
bash tools/run_rt60_contrast_readout_slurm.sh
```

`ACTION=all` 只会提交一个 4-node/16-DCU train job，再提交 random/normal 两个
`afterok` validation jobs；不会提交 held-out test。默认训练和验证时限均为 `23:50:00`，
避免先前集群对超过一天请求的非预期截断。脚本会打印三个 Slurm job ID。

若实际 retry1 目录不在上述位置，只修改 `RT59_PARENT_CHECKPOINT` 为真实绝对路径；不要
复制、重命名或猜测另一份 checkpoint。若目标 RT60 目录已有非空内容，使用一个新的空
目录，不要设置覆盖开关复用失败产物。

## 结果判定

RT60 的主对照是同次保存的 RT59 reference，历史 RT57 gamma=1.66 只作单独历史对照。
`rt60_mechanism_pass` 要求单站 pairwise MAE 和 P95−P05 range absolute error 的差值
95% CI 上界均小于 0，同时 random overall range absolute error、random/normal
non-input MAE/RMSE、observed input、以及规定的 NLL/Brier 均不得退化。

`legacy_full_go` 仍按校正后的 RT59 legacy gates 独立报告。即使 mechanism pass，也不等于
完整空间场目标已经达成，更不构成 test 泛化证据；若失败，记录负结果并保留 RT59，不自动
追加 epoch 或扩大解冻范围。
