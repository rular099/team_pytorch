# FE01 ON V4 产物与核对入口

所有新HPC输出在新run root，原训练、旧评价session、lock、seal和checkpoint只读。
`plan.json`/`analysis_contract.json`/`source_identity.json`冻结源与合同；资源配置及每次真实运行环境另记。
每task独立attempt，失败保留，成功原子发布`stages/STAGE/INDEX/`；`SEAL.json`覆盖全部文件，包括嵌套seal。
父阶段seal SHA随子阶段固定。私有路径环境变更拒绝复用旧run；资源变量可以独立调。

| 阶段 | 核心输出 |
|---|---|
| inventory | checkpoint_inventory.json、source_exports.json、runtime_environment.json |
| compare_existing | cross_encoder_metrics.csv、paired_effects.csv、bootstrap_support.csv、event_robustness.csv、event_effects.csv.gz、first_second_contributions.csv、stratified_effects.csv、interaction_effects.csv、strata_support.csv、spatial_paired_effects.csv、spatial_fields.csv.gz、interaction_contrasts/、additional_endpoints/ |
| covariates | event_covariates.csv、input_station_covariates.csv.gz、request_covariates.csv.gz、decision_manifest.csv.gz、request_manifest.json、grid_manifest.json、metadata_coverage.csv、snr_quality_audit.csv、clock_audit.csv、data_identity.json |
| crossing_truth | crossing_truth.csv.gz、threshold_crossing_audit.csv、label_definition_audit.json |
| smoke/pilot/replay | stores.json索引、每事件/geometry连续station/grid CSV.gz、replay_support.csv、checkpoint_identity.json、metadata_selection.json、smoke_checks.json、latency_frames.csv、worker.json |
| latency | latency_frames.csv、latency_summary.csv、timing_contract.json、实际渲染预览 |
| analyze | 旧协议补物理质量分层、map_readiness_by_frame.csv.gz、event_time_to_usable_map.csv、ttum_pairwise_events.csv、ttum_support.csv、shadow_station_outcomes.csv.gz、shadow_event_summary.csv、shadow_pairwise_events.csv、shadow_support_audit.csv、shadow_area_support.csv |
| render | 每事件20低分辨率帧、MP4/GIF、关键帧、index.html、shadow_station_map.html、frame_sources.json、render_latency.csv；块map_index.html与visuals_index.json |
| pack | review.tar命名见运行手册；full_visuals_index.html全事件入口；review内汇总/合同/来源/12预选案例/完整图源索引；render_delivery_full为实际渲染交付场景补表 |

station行身份：dataset_id/event_id/protocol/geometry/elapsed_time/station_id；另记request_id、输入IDs/顺序/坐标、整数cutoff、实际物理历史与mask指纹、checkpoint SHA/epoch/seed/ON、单位、支持状态和原始MDN logits/mu/sigma。
主误差单位dex，物理坐标为log10(m/s²)。grid没有truth，缺高程点为unsupported_grid_metadata；不造假标签。

完整原始store/视频不进Git。full_artifact_index.json给每个原文件的run-relative路径、字节数、SHA和源seal SHA。
源seal验证未脱敏字节；review复制JSON脱敏后另立REVIEW_MANIFEST，不能回写或重签原seal。
partial pack必须显式给原因，标PARTIAL_WITH_FAILURES，保留失败attempt与job状态，不代表科学scope完成。

本地报告中的真实旧预测统计与合成端到端材料分目录、分seal。合成图醒目标SYNTHETIC，不能用于Japan论文成绩。
正式checkpoint/实际DCU/scheduler结果本地不可见者保留NOT_RUN/NOT_SUBMITTED；源代码存在不等于相应超算阶段已运行。
