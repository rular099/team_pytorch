# 轻量同步设计

目标是跨终端恢复科研任务与供ChatGPT读取报告。Git保存足以理解/修改/定位工作的信息，
大型运行产物通过本地或U盘移动；二者用完整commit、相对路径和SHA-256连接。

## 内容取舍

| 类别 | 策略 | 本轮实施 |
|---|---|---|
| 代码/配置/测试/任务书/交接 | Git | 科研代码基线保持；新增未提交报告脚本、历史任务依据、讲稿与同步工具 |
| 指标与证据 | Git | 聚合/分层CSV、formal metrics、epoch/split/provenance、原来源hash |
| 图像 | 限量Git | RT55/56正文全部21张PNG预览，另有一张小震诊断图；单文件均小于1MiB，重复blob去重 |
| 大逐目标表、raw NPZ/HDF5/权重 | 本地/U盘 | 不上传；逐文件相对路径、大小、SHA、来源工作区及恢复角色入清单 |
| PDF/PPT、SVG与渲染草稿、压缩包 | 通常本地/U盘 | 演示只上传构建代码、讲稿和数值表；小震图SVG不足100KB作为小型矢量证据保留 |
| 缓存、venv、bytecode | 重建 | 不上传，不逐文件当作科研证据备份；环境依赖须按原requirements/audit恢复 |
| 凭据/私有env/本机AI配置 | 私下配置 | 不上传，不读取/公开私有内容；项目申请和会话私有记录也未纳入公开同步 |

默认新文件上限1MiB、单批16MiB。体积不是唯一标准：几KB的密钥也不上传，
已跟踪的历史大报告不因新策略删除或重写。规则只检查本轮新增/修改的暂存文件。
原报告的定量内容不重算，公开副本变量化私有路径、规范Markdown尾空白；CSV原换行与SVG字节按属性保留。
报告脚本仅把字体位置改成当前用户目录及NATIVE_REPORT_FONT环境覆盖，不改统计/模型算法。

## 分支与来源

三个科学工作区的轻量增量放在独立 `sync/rt55-rt56`、`sync/fe01`、`sync/v01`。
本分支负责总索引与工具。原科研分支和原文件不覆盖、不合并、不强推；
新的同步commit的parent固定为对应本地科研基线，未提交内容不会混入另一实验线。
A01本地HEAD落后远端，恢复时使用已核验的较新远端commit，旧本地状态单独保留。
FE02/A01/ON没有本轮需再上传的代码增量时，直接引用原远端commit。

完整恢复入口：[TERMINAL_HANDOFF](TERMINAL_HANDOFF.md)。机器可读索引：
[terminal_manifest.json](terminal_manifest.json)，原工作区对照：
[local_worktrees_20261102.json](local_worktrees_20261102.json)。
工作区外任务书归档在workspace_sources/，仅作带来源的历史材料，不能重新激活其旧P0或启动指令。

## 日常同步

```bash
# 在总入口checkout中检查某条实验线；命令只读，不会自动add/push
python tools/workspace_sync.py inventory --repo ../team_pytorch_fe02
# 人工选择本轮代码/记录/轻量结果，逐路径暂存，保留raw产物
git -C ../team_pytorch_fe02 add -- path/to/code.py path/to/report.md
python tools/workspace_sync.py check-staged --repo ../team_pytorch_fe02
git -C ../team_pytorch_fe02 diff --cached --check
# 验证通过后正常commit/push该线，协调者再记录精确SHA与本地大产物清单
```

不使用无审查的git add -A。程序不会自动提交、推送、合并或删除。
同一快照分支如未来两终端均产生提交，应先fetch并审查分叉；同步工具不使用force解决分叉。
需引入公共记忆到科学工作分支时仍遵循既有单独审阅流程，本轮快照不替代该审阅。

## 可验证的边界

每份公开源文件都有原SHA与发布SHA，区分路径脱敏/格式规范化/字体路径适配。
原封包MANIFEST与已脱敏轻量子集不是同一字节集合，不能直接套旧seal。
本轮验证传输内容与源表语义，不声称重载历史权重或重新运行所有科学评价。
当前本机Git原生SSH/HTTPS传输受配置/DNS/密钥限制，发布通过已授权GitHub连接器完成，
远端commit/tree/ref与本地Git对象逐项验证。其他终端仍需配置自己的Git认证与网络。
