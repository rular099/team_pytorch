# FE01-EVAL1：本地和超算各做什么

**本地已完成九组CSV的真实再分析；接下来超算只补验证和评价，不训练。** 本轮没有提交任何作业。

现在使用直接提交版：`artifacts/fe01/fe01_eval1_hpc_submit_20261005.tar.gz`及同名`.sha256`。这两个文件替代此前的评价交付包。包内含新增评价代码、提交脚本、配置、运行手册、原始请求元数据、两个legacy原config快照及已填路径的私有`cluster_zb.env`。不包含checkpoint/HDF5，不覆盖原115个训练源码文件，不重签旧lock，也不要求再拷1GB原权重包。

| 在哪里 | 要做的事 |
|---|---|
| 本地电脑 | 将这两个交付文件拷到超算已有的`team_pytorch_fe01`项目根目录；本地真实报告在`reports/fe01_eval1_review_20261005/` |
| 超算登录节点 | 解压新包、执行`bash scripts/fe01_review/submit.sh 阶段名`提交作业、查看`squeue/sacct` |
| 超算计算节点 | 下面的sbatch脚本自动初始化module/conda zb，并运行其指定的CPU或DCU任务 |
| 本地电脑 | 下载最后的轻量证据包供下一轮整理分析；完整数值输出留在超算新review目录 |

## 先在超算登录节点执行

先把**新的tar.gz和.sha256两个文件**拷到你此前一直使用的**超算FE01项目根目录**，再在该目录执行下面命令。包里的私有env已经从你之前的launcher/config填好绝对路径，输出使用新run ID `fe01_eval1_direct_20261005`。下面不用手动激活conda；计算节点上的作业脚本会source conda.sh后激活zb。

```bash
sha256sum -c fe01_eval1_hpc_submit_20261005.tar.gz.sha256
tar -xzf fe01_eval1_hpc_submit_20261005.tar.gz
sha256sum -c artifacts/fe01/eval1_overlay.sha256
bash scripts/fe01_review/submit.sh identity
```

**最后一行会直接调用sbatch提交第一份CPU作业。** 脚本自动读取同目录`cluster_zb.env`、创建新输出目录和logs，返回`SUBMITTED identity JobID=...`。记下这个Job ID，用`squeue -j 实际JobID`看状态。第一份作业完成前先不要提交后续阶段。无需生成txt或复制里面的命令。

`submit.sh`是你在登录节点执行的提交入口；`*_job.sbatch`是它提交给计算节点的实际作业。提交入口优先使用你已有的`~/.conda/envs/zb/bin/python`，无需在shell里执行conda activate；该Python只检查少量JSON并调用sbatch，不读取波形或加载torch。每次调用只提交指定阶段，记录实际Slurm返回值到`jobs_manifest.json`；它不会自动连续提交其他阶段。原打印入口仍保留作只读准备工具，本手册不再使用它。

同一阶段重复执行会停止，不会重复提交或覆盖结果。需要重做时在私有env中设置新的`REVIEW_RUN_ID`，从identity重新开始。若旧包已生成过`fe01_eval1_20261005`目录，新包的`fe01_eval1_direct_20261005`使用另一目录。旧验证报告的源码SHA不能沿用到这份新代码。

`FE01_TRAIN_OUTPUT_ROOT`指原训练输出，仅只读；`FE01_REVIEW_OUTPUT_ROOT/REVIEW_RUN_ID`才是新评价输出，默认在FE01项目内。`RT55_CHECKPOINT`/`RT61_CHECKPOINT`可以指各自原实验目录：inventory读取该目录各pth的内部epoch，只接受预选32/8；若同epoch有不同state则BLOCKED，不能按loss再选一个。原模型的external encoder也必须在checkpoint记录路径，不能随意换成另一份文件。

## 提交顺序：一次只推进已经过门的阶段

下面所有命令都在**超算登录节点、FE01项目根目录**执行。等待前置阶段完成后，再执行需要的下一行；不要一次粘贴整张表。日志在私有env声明的`FE01_REVIEW_OUTPUT_ROOT/REVIEW_RUN_ID/logs/`。

| 直接执行的命令 | 前置条件 | 提交内容 |
|---|---|---|
| `bash scripts/fe01_review/submit.sh identity` | 新run ID，首先执行 | CPU权重身份核验、共同请求准备 |
| `bash scripts/fe01_review/submit.sh verify` | identity完成 | 身份通过的模型，各1卡，小样本前向核验 |
| `bash scripts/fe01_review/submit.sh reference` | identity完成，可与verify并行 | CPU共同train-only参照 |
| `bash scripts/fe01_review/submit.sh legacy_fixed` | verify完成 | 核验通过的RT55/RT61固定时刻复评 |
| `bash scripts/fe01_review/submit.sh random` | verify完成 | 核验通过模型的共同随机时刻补评 |
| `bash scripts/fe01_review/submit.sh long` | verify完成 | 核验通过的TEAM/EQT/旧系统，按原生能力补评 |
| `bash scripts/fe01_review/submit.sh replay` | verify完成；reference完成后可附参照图 | 核验通过的seed42模型和旧系统，3例逐秒回放 |
| `bash scripts/fe01_review/submit.sh analyze` | reference和verify完成 | CPU原CSV加台站诊断 |
| `bash scripts/fe01_review/submit.sh pack` | 所需阶段已提交；最后执行 | 汇总、SHA校验、轻量打包 |

提交脚本会检查前置产物，并自动跳过缺权重、身份未通过或前向核验未通过的模型；打印`SKIP 模型 原因`。所有候选都不合格时不提交空数组。下游作业仍会在计算节点检查真实权重和源码SHA。你不用填写数组索引。

pack带有此前已提交Job ID的`afterany`依赖，会等这些作业结束（包括失败）后才运行。提交pack后脚本不再接受其他阶段，避免遗漏后来提交的证据。各模型成功由核验报告确定；数组内某个模型失败不阻塞已经通过的其他模型。实际scheduler依赖和返回值保存在`jobs_manifest.json`，`SUBMITTED`不等于`COMPLETED`。

1. **identity：CPU身份和共同请求准备。** 首先执行表格的identity命令。它顺序生成`identity/checkpoint_inventory.json`、`identity/frozen_checkpoint_manifest.json`、`requests/`。九组checkpoint缺失只阻塞对应模型，不阻塞其他模型的请求计划；HDF5或原115源码身份不一致则阻塞计划。没有缺项会写IDENTITY_PASS，但它仍不是forward PASS。

2. **verify：每模型小样本前向检查。** identity结束且请求计划完成后，执行verify命令。每个数组元素只检查一个固定权重，1节点1DCU；正常可并行8个。失败只影响该模型，不会自动补跑或调容差。查看`verification/<run_id>/verification.json`：只有PASS才允许该模型进入下一步。

3. **legacy_fixed：旧完整系统共同固定人口复评。** 脚本只对通过verify的RT55/RT61提交。没有epoch32时RT55留下BLOCKED；RT61仍可单独运行。九组FE01不完整重跑固定时刻，复用其已核验CSV。

4. **random / long：缺失时刻补评。** verify完成后分别执行这两个提交命令。random用3930draw的共同原manifest，重复快照只算一次。long默认TEAM/参考40、90和EQT40；EQT90写unsupported_history，PhaseNet整项N/A。不会缩短历史或开启rolling。

5. **reference：CPU共同train-only参照。** identity的数据身份通过后可与GPU评价并行提交。只读9084训练事件的146299最终标签，不读取validation/test拟合。输出`reference/train_only_reference.json`、实际训练journal曝光和缺项。无需DCU，也不训练任何模型。

6. **analyze：CPU九组旧CSV加台站诊断。** reference和verify完成后执行analyze命令。已有纯CSV诊断本地已经完成；此阶段的新增价值是共同oracle参照、跨事件台站残差及训练曝光，不为每张图重读14GB原始目录。

7. **replay：3例有限逐秒回放。** 执行replay命令，脚本选通过verify的seed42三模型及参考；reference先完成可附共同no-site图。3例×1–20秒×normal/random×自然/固定S0；真实站query与网格分开。输出逐帧数据、1/5/20秒图片、latency和独立快照复验；本轮不生成动画、不跑全国网格。

8. **pack：校验并轻量打包。** 计划执行的阶段都已提交，或已经记录BLOCKED后，最后执行pack命令。它等待已提交作业结束，检验已生成文件SHA，输出可用认证模型的系统比较、各scope实际分母和未完成门，不把缺失结果写成PASS。`return_package/fe01_eval1_evidence.tar.gz`供下载；完整预测CSV、训练标签和帧数据仍留review目录。

下面表格仅供查日志时对应模型，**提交时不需要使用这些数字**。脚本自动根据身份/前向核验结果选择数组成员。这里的数字只用于本轮新脚本，与旧训练数组完全无关。

| 数组ID | 固定模型 | epoch | 使用场景 |
|---:|---|---:|---|
| 0 | TEAM seed42 | 11 | verify、random、long、replay |
| 1 | TEAM seed43 | 11 | verify、random、long |
| 2 | TEAM seed44 | 11 | verify、random、long |
| 3 | PhaseNet seed42 | 12 | verify、random、replay |
| 4 | PhaseNet seed43 | 10 | verify、random |
| 5 | PhaseNet seed44 | 12 | verify、random |
| 6 | EQT seed42 | 10 | verify、random、long、replay |
| 7 | EQT seed43 | 10 | verify、random、long |
| 8 | EQT seed44 | 10 | verify、random、long |
| 9 | RT55完整系统 | 32 | verify、legacy_fixed、random、long、replay |
| 10 | RT61增强系统 | 8 | verify、legacy_fixed、random、long、replay |

例如日志中的数组ID 0、3、6分别对应TEAM42、PhaseNet42、EQT42，各自1节点1卡。CPU作业不是数组。每个stage的实际run IDs和Slurm数组参数都保存在`jobs_manifest.json`。

## 看状态、下载哪些东西

把sbatch返回的实际Job ID记下来。你自己执行下面只读命令；Codex不替你远程执行。

```bash
source scripts/fe01_review/cluster_zb.env
squeue -u "$USER"
sacct -j <逗号分隔的实际JobID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P \
  > "$FE01_REVIEW_OUTPUT_ROOT/$REVIEW_RUN_ID/sacct_status.txt"
```

在pack前保存sacct，才会收入证据包。`jobs_manifest.json`记录实际提交回执，作业是否COMPLETED仍须由sacct和产物共同确认。首批失败先看对应`.err`、`failure.json`、`verification.json`或checkpoint_inventory；脚本会跳过该模型下游计算。

下载`return_package/fe01_eval1_evidence.tar.gz`、`return_package/package.json`以及实际`sacct_status.txt`即可供ChatGPT审阅。如果需要检查某个前向差异，再从review目录取该模型原始预测CSV；不用再次下载所有权重和14GB旧训练结果。轻量包没有视频就不声称有视频。

请保留`FE01_REVIEW_OUTPUT_ROOT`新输出。旧RT55/RT61输出、旧FE01训练目录、split和protocol.lock不改。验证失败后需要改代码时重新生成新评价source pin并使用新run ID，不沿用旧verification PASS。
