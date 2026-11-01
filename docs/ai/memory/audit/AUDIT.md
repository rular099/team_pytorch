# 初始化审计与证据边界（2026-11-01）

Repository：rular099/team_pytorch；协调分支chore/project-memory-bootstrap。
实现base：ba4fa7740d9fde5fde87a7b0ae0397037209baf4，是FE01/FE02/A01的已核验共同祖先。
仅沿用该base既有代码；没有合并其他科研分支。

## 已检查的范围

AGENTS、PROJECT_CONTEXT、docs/ai/README、SESSION_SUMMARY、Git所有本地/remote-tracking分支及近期提交、
tags（本地无tag），RT55/56与query诊断、RT57–RT61结果交接/指标门、FE01训练/EVAL1报告、
FE02协议/最新恢复交接/输出和cache路径、A01新结果、ON新协议/CPU结果/inventory恢复、V01 closure。
早期stage2报告与仓库外handoff_docs作为历史导航，不作为当前全局任务。
远端通过GitHub插件GET独立确认24个branch，完整列表与时间见remote_refs.json。
未读取全部ChatGPT历史、未重新下载权重/大型NPZ、未连接超算或读取私有env。

## 避免过时状态的关键发现

| 来源 | 已确认范围 | 纠正/处理 |
|---|---|---|
| 主目录8bde65c | 9月2日RT55/RT56上下文；大量未跟踪产物 | 保留；不当成项目最新HEAD |
| FE01 ba4fa77 | 10月6日EVAL1：11 identity、8 forward PASS、3 TEAM BLOCKED | 该阻挡属旧来源；A01/ON独立记录 |
| A01本地236f727 | 10月15日等待评价/pack回传 | 已被远端29ffb20的10月21日完整结果取代 |
| A01远端29ffb20 | 五项seal、16份验证导出、中立结果报告 | 记verified_complete仅指封包；独立解释待审阅 |
| ON远端50672e1 | 10月25日inventory路径修复；此前真实CPU结果已发布 | 生产模型回放未认证；合成图不作Japan新结果 |
| FE02 42b3d54 | 10月14日M组5/12轮；RNG兼容层交付 | 真实恢复未回传，其他组unknown |
| V01 d8f2436 | 10月6日工程闭环结果 | 有限科学解释待审阅；不是整个仓库最新科研工作 |
| RT59/60/61 | 点指标进展与联合门失败并存 | 分别保留22/29、8/14、13/14及0/4，不改成功标准 |

## 本地/远端与路径问题

旧worktree注册仍指向迁移前绝对路径，常规git -C失败。本轮以主仓库Git对象和
显式git-dir/work-tree只读核验旧HEAD/status/diff，新建team_pytorch_memory独立worktree。
没有repair/prune旧注册，没有更改旧工作树的branch、HEAD、tracked diff或未跟踪清单。
原主目录351条、FE01 64条、V01 176条状态项均保留；A01、FE02原状态干净。
状态基线不包含每个未跟踪大文件的独立初始字节hash，因此不声称完成全量大产物字节复核。

本地HTTPS git ls-remote因DNS不可用失败，GitHub连接器可读写仓库。两个新FE01远端commit
不在本地对象库；保存五份原文和各自Git blob SHA以便离线核验，原相对链接仍指向来源目录。
这是2026-11-01的一次连接器读取记录，不能代替未来在线HEAD核验或原科学大产物重算。

## 并发写入审计

现有FE01/FE02 launcher分别使用OUTPUT_ROOT；FE02带run锁与清单注册锁，FE01基线cache在本线根共享。
两个WEIGHTS_ROOT都可能写pretrained_manifest；变量名不同不证明实际路径不同。
实际部署env/路径未读取，故deployment_not_verified。新增只读resources检查和命名规范，
检查根包含/symlink/write-read冲突；没有修改在跑作业或科研launcher。

当前FE01 HEAD就是FE02分叉基点，所以相对merge-base没有双方同时修改文件；但FE02自身改过共享架构，
仍须单独架构审阅。临时仓库已模拟双方改同一gemini_models.py并识别风险，未执行自动merge。

## 自动检查的边界

检查结构化字段、有效状态、full SHA/commit存在性、固定commit路径、remote snapshot blob、
重复ID、快照/生成区冲突、链接、过期核验和写入范围。不能自动判断所有自然语言科学冲突；
人工仍须核对协议/原产物。所有科学结论从可见报告人工提炼，未凭目录名或指标文件名生成。
