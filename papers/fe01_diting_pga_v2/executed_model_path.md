# 实际模型路径核验

仓库 rular099/team_pytorch；论文分支 paper/fe01-diting-pga-manuscript-v2；论文基点 560a29d8c82ba634549ed886840b2a2a539e0114；执行来源 50672e131d876a604432b283162d5b0189d40cf0；结果来源 b366434d2cceb55d188576ce1084f571e09da097。所有文件通过固定Git对象读取，未导入模型或加载权重。

验证的路径为 dataset/sampler → 截止和存储掩码 → 物理波形与坐标 → 前端 → 目标读出 → MDN/损失 → 导出/回放。历史 FE01_PROTOCOL 只作名义协议，实际执行以完整配置和源分支为准。

## fe01/data.py

- [read_event，63–146](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/data.py#L63)。
- [geometry，149–173](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/data.py#L149)。
- [prepare_sample，191–243](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/data.py#L191)。
- [TrainingDataset，246–269](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/data.py#L246)。
## fe01/windows.py

- [clock，43–46](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/windows.py#L43)。
- [build_window，49–106](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/windows.py#L49)。
## fe01/model.py

- [shape_and_scale，31–54](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/model.py#L31)。
- [FE01FullModel，57–110](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/model.py#L57)。
- [build_model，113–157](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/model.py#L113)。
- [_encode_station_waveform，87–110](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/model.py#L87)。
## fe01/engine.py

- [evaluation_rows，76–140](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/engine.py#L76)。
- [loss，395–404](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/engine.py#L395)。
- [train，407–570](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/engine.py#L407)。
- [export_evaluation，585–619](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/engine.py#L585)。
## fe01/metrics.py

- [decode，7–12](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/metrics.py#L7)。
- [score_rows，37–60](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/metrics.py#L37)。
## fe01/extractors.py

- [PhaseNetBottleneck，50–71](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/extractors.py#L50)。
- [EQTSharedTrunk，74–88](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/extractors.py#L74)。
- [StationFrontend，138–174](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01/extractors.py#L138)。
## fe01_a01/model.py

- [build_model，24–69](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01/model.py#L24)。
## gemini_models.py

- [MixtureOutput，331–369](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L331)。
- [CrossAttentionReadout，700–843](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L700)。
- [forward，322–329](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L322)。
- [forward，360–369](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L360)。
- [forward，386–390](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L386)。
- [forward，415–441](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L415)。
- [forward，452–460](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L452)。
- [forward，493–500](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L493)。
- [forward，533–539](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L533)。
- [forward，566–577](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L566)。
- [forward，680–697](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L680)。
- [forward，814–843](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L814)。
- [forward，877–887](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L877)。
- [forward，983–1050](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L983)。
- [forward，1124–1191](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1124)。
- [forward，1289–1433](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1289)。
- [forward，1455–1471](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1455)。
- [forward，1551–1656](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1551)。
- [forward，1677–1699](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1677)。
- [forward，1730–1736](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1730)。
- [forward，1855–2059](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L1855)。
- [forward，2088–2162](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2088)。
- [forward，2268–2324](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2268)。
- [forward，2406–2452](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2406)。
- [forward，2517–2566](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2517)。
- [forward，2649–2698](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2649)。
- [forward，2722–2728](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2722)。
- [forward，2742–2752](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2742)。
- [forward，2771–2779](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2771)。
- [forward，2785–2795](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2785)。
- [forward，2806–2812](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2806)。
- [forward，2826–2827](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2826)。
- [forward，2902–2955](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L2902)。
- [forward，3009–3024](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3009)。
- [forward，3084–3121](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3084)。
- [forward，3140–3141](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3140)。
- [forward，3186–3193](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3186)。
- [forward，3212–3215](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3212)。
- [forward，3314–3319](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3314)。
- [forward，3511–3580](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3511)。
- [forward，3720–3766](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3720)。
- [forward，3826–3864](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3826)。
- [forward，3877–3883](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L3877)。
- [_station_memory_for_readout，5329–5378](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L5329)。
- [_pga_point_mean_from_output，5651–5657](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L5651)。
- [forward，5858–6759](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/gemini_models.py#L5858)。
## fe01_a01_compare/map_readiness.py

- [frame_metrics，7–27](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/map_readiness.py#L7)。
- [gate，30–40](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/map_readiness.py#L30)。
- [times_to_ready，43–77](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/map_readiness.py#L43)。
## fe01_a01_compare/shadow.py

- [probabilities，9–24](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/shadow.py#L9)。
- [classify，27–69](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/shadow.py#L27)。
## fe01_a01_compare/clocks.py

- [fifo，32–48](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/clocks.py#L32)。
## fe01_a01_compare/latency.py

- [benchmark，10–83](https://github.com/rular099/team_pytorch/blob/50672e131d876a604432b283162d5b0189d40cf0/fe01_a01_compare/latency.py#L10)。

## 已证实分支

四模型的 station_context_mode 为 off，pga_use_event_context 为 false，pga_readout_mode 为 target_cross_attention，pga_readout_layers 为4，readout_first_residual 为 true。绝对坐标启用；相对坐标、RoPE、Vs30、目标时间池化与 PGA 时间残差均未启用。十一维物理量与时长仍在实际输入中。图1不把未激活的台站自注意力接到PGA路径，也不把辅助震级/位置预测输入PGA。

shape_and_scale对有效掩码计算均值、尺度及有效时长，非有效样本不进入统计；前向输入没有最终标签。read_event只读取排他cutoff上界之前的波形，但允许读取最终PGA作监督；prepare_sample将输入、标签分开。冻结骨干在eval/no_grad边界内执行，投影/适配与读出可训练。

## 身份与表示边界

DiTing完整模型的历史配置与接口审计只用于确认本文候选身份；不取其他模型条件结果补齐论文。完整设置见 complete_model_settings.json，检查点轮次/校验值见固定结果 metadata/checkpoint_inventory.json。原始私有config摘要与去路径配置副本的字节摘要不同，本轮只核验可见表示与实际标志，未声称重新读取私有原件。精确预训练论文、语料与许可未查明。

## 评价与代价

station gate的阈值为MAE≤b、|bias|≤b/2、至少90%目标误差≤2b；field gate再约束centered RMSE和range_error≤b。三连续帧第三帧才确认，确认后已知不通过另计。shadow及时要求提前量严格为正且≥tau；未受影响的完整负例才进入FPR。FIFO逐事件逐模型计算，含渲染队列中的后帧数值已受前帧渲染阻塞。

## 空间指标补充核验

- [fe01_a01_compare/spatial_statistics.py::field_intervals 8–37](https://github.com/rular099/team_pytorch/blob/560a29d8c82ba634549ed886840b2a2a539e0114/fe01_a01_compare/spatial_statistics.py#L8)。

此处核读论文基点的定义文件，不把它们混写为新回放执行版本。range为P95−P05；等距目标对采用震中距之差≤5 km的既定规则；level/shape/field使用均值误差平方、中心化误差平方均值、原误差平方均值。场bootstrap对支持单元等权，并保留空单元和有效重复数。
