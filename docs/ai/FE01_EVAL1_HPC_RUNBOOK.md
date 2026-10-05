# FE01-EVAL1：本地和超算各做什么

**本地已完成九组CSV的真实再分析；接下来超算只补验证和评价，不训练。** 本轮没有提交任何作业。

本地交付包为`artifacts/fe01/fe01_eval1_hpc_20261005.tar.gz`及同名`.sha256`。包内只有新增评价代码、配置、运行手册、原始请求的元数据、两个legacy原config快照及已填路径的私有`cluster_zb.env`。不包含checkpoint/HDF5，不覆盖原115个训练源码文件，不重签旧lock，也不要求再拷1GB原权重包。

| 在哪里 | 要做的事 |
|---|---|
| 本地电脑 | 将这两个交付文件拷到超算已有的`team_pytorch_fe01`项目根目录；本地真实报告在`reports/fe01_eval1_review_20261005/` |
| 超算登录节点 | 解压新包、检查环境路径、打印命令、手动执行打印的`sbatch`、查看`squeue/sacct`；不跑网络前向 |
| 超算计算节点 | 下面的sbatch脚本自动初始化module/conda zb，并运行其指定的CPU或DCU任务 |
| 本地电脑 | 下载最后的轻量证据包供下一轮整理分析；完整数值输出留在超算新review目录 |

## 先在超算登录节点执行

先进入你此前一直使用的**FE01项目根目录**，即包和`scripts/`所在目录。包里的私有env已经从你之前的launcher/config填好绝对路径。下面不用手动激活conda；作业脚本会source conda.sh后激活zb。

```bash
sha256sum -c fe01_eval1_hpc_20261005.tar.gz.sha256
tar -xzf fe01_eval1_hpc_20261005.tar.gz
sha256sum -c artifacts/fe01/eval1_overlay.sha256
export FE01_REVIEW_ENV="$PWD/scripts/fe01_review/cluster_zb.env"
source "$FE01_REVIEW_ENV"
bash scripts/fe01_review/print_submit_commands.sh "$FE01_REVIEW_ENV" > fe01_eval1_submit_commands.txt
cat fe01_eval1_submit_commands.txt
```

打印器只创建新review根目录、logs和`jobs_manifest.json`，**没有执行sbatch/srun**。默认DRY_RUN=1；DRY_RUN=0直接拒绝。不要为了再次打印而复用非空run ID；首次打印后保留文件，以后从该文件复制命令。确实要重做失败阶段时使用全新的REVIEW_RUN_ID，不能覆盖前一份证据。

`FE01_TRAIN_OUTPUT_ROOT`指原训练输出，仅只读；`FE01_REVIEW_OUTPUT_ROOT/REVIEW_RUN_ID`才是新评价输出，默认在FE01项目内。`RT55_CHECKPOINT`/`RT61_CHECKPOINT`可以指各自原实验目录：inventory读取该目录各pth的内部epoch，只接受预选32/8；若同epoch有不同state则BLOCKED，不能按loss再选一个。原模型的external encoder也必须在checkpoint记录路径，不能随意换成另一份文件。

## 提交顺序：一次只推进已经过门的阶段

每个小标题的名字与打印文件一致。这里说“复制命令”是复制打印出的**完整sbatch行**并在登录节点按回车，不是执行Python。日志在`$FE01_REVIEW_OUTPUT_ROOT/$REVIEW_RUN_ID/logs/`。

1. **identity：CPU身份和共同请求准备。** 首先只复制`identity`下面那一行。它顺序生成`identity/checkpoint_inventory.json`、`identity/frozen_checkpoint_manifest.json`、`requests/`。九组checkpoint缺失只阻塞对应模型，不阻塞其他模型的请求计划；HDF5或原115源码身份不一致则阻塞计划。没有缺项会写IDENTITY_PASS，但它仍不是forward PASS。

2. **verify：每模型小样本前向检查。** identity结束且请求计划完成后，复制`verify`行。每个数组元素只检查一个固定权重，1节点1DCU；正常可并行8个。失败只影响该模型，不会自动补跑或调容差。查看`verification/<run_id>/verification.json`：只有PASS才允许该模型进入下一步。

3. **legacy_fixed：旧完整系统共同固定人口复评。** 只对通过verify的RT55/RT61执行`legacy_fixed`。没有epoch32时RT55必须留下BLOCKED；RT61仍可单独运行。九组FE01不完整重跑固定时刻，复用其已核验CSV。

4. **random / long：缺失时刻补评。** 每个模型verify通过后可以分别执行这两行。random用3930draw的共同原manifest，重复快照只算一次。long默认TEAM/参考40、90和EQT40；EQT90写unsupported_history，PhaseNet整项N/A。不会缩短历史或开启rolling。

5. **reference：CPU共同train-only参照。** identity的数据身份通过后可与GPU评价并行提交。只读9084训练事件的146299最终标签，不读取validation/test拟合。输出`reference/train_only_reference.json`、实际训练journal曝光和缺项。无需DCU，也不训练任何模型。

6. **analyze：CPU九组旧CSV加台站诊断。** reference完成后执行`analyze`。已有纯CSV诊断本地已经完成；此阶段的新增价值是共同oracle参照、跨事件台站残差及训练曝光，不为每张图重读14GB原始目录。

7. **replay：3例有限逐秒回放。** 对通过verify的seed42三模型及参考提交`replay`行；reference先完成可附共同no-site图。3例×1–20秒×normal/random×自然/固定S0；真实站query与网格分开。输出逐帧数据、1/5/20秒图片、latency和独立快照复验；本轮不生成动画、不跑全国网格。

8. **pack：校验并轻量打包。** 上面计划执行的作业都结束，或已记录BLOCKED后，复制`pack`行。它检验已生成文件SHA、输出可用认证模型的系统比较、各scope实际分母和未完成门，不把缺失结果写成PASS。`return_package/fe01_eval1_evidence.tar.gz`供下载；完整预测CSV、训练标签和帧数据仍留review目录。

没有认证的模型可以从数组命令中去掉；下面表格解释每个数字。若只有RT61可用，将`--array=9,10%8`改为`--array=10`即可。这里的数字只用于本轮新脚本，与旧训练数组完全无关。

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

例如`--array=0,3,6%3`表示三份并行作业：TEAM42、PhaseNet42、EQT42，各自1节点1卡；不是三份模型在同一Python进程。CPU作业不是数组。sbatch文件里的Python是在获准的计算节点内执行，不是在登录节点直接启动。

## 看状态、下载哪些东西

把sbatch返回的实际Job ID记下来。你自己执行下面只读命令；Codex不替你远程执行。

```bash
squeue -u "$USER"
sacct -j <逗号分隔的实际JobID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P \
  > "$FE01_REVIEW_OUTPUT_ROOT/$REVIEW_RUN_ID/sacct_status.txt"
```

在pack前保存sacct，才会收入证据包。`jobs_manifest.json`只是计划，没有实际scheduler记录时不能证明COMPLETED。首批失败先看对应`.err`、`failure.json`、`verification.json`或checkpoint_inventory；不要继续提交该模型下游计算。

下载`return_package/fe01_eval1_evidence.tar.gz`、`return_package/package.json`以及实际`sacct_status.txt`即可供ChatGPT审阅。如果需要检查某个前向差异，再从review目录取该模型原始预测CSV；不用再次下载所有权重和14GB旧训练结果。轻量包没有视频就不声称有视频。

请保留`FE01_REVIEW_OUTPUT_ROOT`新输出。旧RT55/RT61输出、旧FE01训练目录、split和protocol.lock不改。验证失败后需要改代码时重新生成新评价source pin并使用新run ID，不沿用旧verification PASS。
