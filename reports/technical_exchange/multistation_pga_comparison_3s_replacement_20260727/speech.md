## Slide 1: 文献与本项目的数值对比——本项目统一为 3 s

这一页把本项目的比较时间统一到首个有效 P 波后的固定 3 秒。左侧 TEAM 数字仍是论文公开的日本 test 结果，但 TEAM 采用从首个 P 波后 0.5 秒开始、每 0.1 秒更新、最长约 25 秒的概率预警流程，因此它并不是一个固定 3 秒快照。本项目 train 行只显示拟合容量，真正需要关注的是 validation。

右侧 QuakeFormer 的这组 R² 和残差标准差来自 Figure 3 的 forecasting 任务。该任务输入事件参数和场地信息，不使用实时波形，所以没有“多少秒”的定义。这里用本项目 rt48 的固定 3 秒结果作量级参照，但两者仍不是同任务、同数据集或同划分，不能理解成排行榜。

---

注意点：
- 重点：本项目左右两处均为 rt48@3 s；QuakeFormer forecasting 无固定秒数。
- 画面引导：先看左表 train—validation 差距，再看右表 held-out 条件与本项目 validation。
- 补充：QuakeFormer 另有 0–17 s 的 EEW 时间曲线，但不是本页采用的 Figure 3 forecasting 数字。
