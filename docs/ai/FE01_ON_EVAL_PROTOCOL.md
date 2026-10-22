# FE01 ON V4 科学与执行合同

权威任务书与固定JSON均存于 `configs/fe01_a01_compare/`，输入规范是任务资料，不冒充已执行结果。
基点29ffb20；新实现/提交分支 `exp/fe01-on-warning-map-evaluation`。所有当前数据限validation、seed42。

## 模型、人口与估计量

四ON selected固定DiTing12、PhaseNet12、EQT10、TEAM11；epoch12仅复用已有导出作稳健性，不重复密集回放。
旧RT55/RT61系统不参与排名。新源码独立计SHA；旧115、A01核心/lock/旧tests/docs、旧seal不变。
checkpoint先按原文件SHA核字节，再核内部epoch、config/lock、实际state与encoder；TEAM训练态不与初态比较。

`legacy_snapshot_v1`：旧1310事件、194265请求、noninput139440。所有原身份字段严格相同才配对，不取交集。
CSV用C解析器round_trip；统计只读所需列，重复input字符串共享内存，完整MDN保留在原文件及新回放。
所有导出先匹配固定SHA和已发布source/revision/plan门，不重新签名旧门。

主MAE仍为normal/random×1/3/5/10/20的十cell等权；cell内目标等权。新副端点为第一秒两cell、早期1/3/5秒六cell。
差值均DiTing减PhaseNet/EQT/TEAM。RMSE先平均cell MSE再开方，每次bootstrap对两个模型分别开方再相减。
R²/slope/P90/P95标为pooled，不能平均cell分位数或R²冒充pooled；低真值方差写NA。
NLL/CRPS/Brier复用原精确MDN评分；95%和68%区间来自混合分布分位数。68%宽度补表单列。
覆盖率和宽度不能单独机械判优，不调sigma。

事件簇为dataset/event，2000 draws、seed20261007，同一事件全部站点/时间/几何同行。
逐事件胜率、贡献、LOO与目标/cell主指标分开；贡献是代数分解，不是因果解释。
场指标保留原>=5目标、field等权定义，复用严格验SHA的原field表。稀疏空cell/有效draws如实导出。
事件区间不包含重训seed或验证选轮的不确定性；新端点/多重探索默认未校正，<20事件描述性。

## 分层、交互与信息可见性

所有边界/开闭侧见JSON和 `stratification.py`，Unknown保留。震级标度没有证据即catalog native/UNKNOWN，不称Mw。
震中距、最近input距离、凸包内外/退化、跨度、方位缺口、实际K、真实缺测、有效P后秒数和SNR分别报告。
不把native左padding算成物理缺测；合法零不等于缺测。

SNR噪声窗为[P−5s,P−1s)，信号窗为[P,min(P+1s,current)]，信号终点按采样时钟包含，原数据cutout仍排他。
仅共同有效三分量采样；分别至少2s/.5s、覆盖>=80%；先减噪声均值再比含噪观测能量。
零噪声、低于原存储数值分辨率的噪声能量、缺窗、可核验clipping与时钟问题不造高SNR；decision可算站不足80%即Unknown。
P前补零从来不算噪声；未来采样不参与质量计算。

四交互仅time×K、time×SNR、truth PGA×nearest input、magnitude×epicentral distance。
先冻结支持，再看效应；除条件cell表外，`interaction_contrasts.csv`提供固定参考水平的difference-of-differences。
交互对比在四组共同geometry/time/K域标准化（排除交互因素本身），不是原十cell权重。
四组共同事件簇重采样，保留NA与稀疏draw丢弃。缺SNR时对应对比不可识别，不以显著性差代替交互检验。

目录震源/震级、真实最终PGA、完整目标记录、阈值越过时刻均是回顾性评分信息；不进前向/归一化/输入选择。
CPU真值模块按已追踪的NEZ向量范数峰值定义复算标签，完整记录只在此模块读取。
记录内首次越阈不是最大峰值时刻。左端已越阈、越阈前缺测、clipping、标签不一致/边界歧义保留Unknown或删失。
源头完整记录去均值/重采样及目录拾取的在线可用性仍为UPSTREAM_CAUSALITY_UNKNOWN，不能认证连续实时触发。

## 新回放、网格与缓存

`fixed_query_replay_v4`：原1310锁定事件，固定真实目标全集，首P后逐秒1–20s，两种geometry。
所有模型用同一逐帧输入计划；random每事件固定优先级/K规则，不随帧独立重抽；成为input的query继续输出。
`fixed_never_input_targets`按共同1–20秒输入并集回顾性定义；current_noninput仅描述，不定义TTUM。
新请求有独立hash，不能伪装成旧194265请求。

原native_prefix/预处理/ON原样复用，原RT55入口不变。合法短窗之外写NA，不用rolling或压缩时间轴。
每worker只加载一个模型一次；raw前缀每事件一次。缓存token包含checkpoint/mode/cutoff/IDs/坐标，实际wave/mask亦核指纹。
只在同截止station/grid分块间复用编码。原snapshot重叠点核完整MDN；输入不同单列manifest差异。

网格用声明的局部等距近似投影（非面积认证），10km基准、台站范围外20km、最多4096点，整数倍增间距。
第三坐标由生产链确认stationheight(m)/1000，即km高程。仅台站凸包内线性估算静态高程，显式metadata_estimated；外部点缺高程为NA。
退化/缺元数据事件交站点支撑展示。零高程不偷偷进正式计算。无连续域真值，无独立共同面积支撑则面积NA。
全部帧预测同一个最终PGA，图色标/等值线固定；真值/震中图注为回顾性。直接站点query误差不与网格插值混合。

## TTUM、shadow与时钟

TTUM是station_supported_time_to_usable_map，离线研究量，engineering_acceptance=null。
完整报告b=.15/.20/.25dex：MAE<=b、abs(bias)<=b/2、90%绝对误差<=2b；field再加centered RMSE<=b、range误差<=b。
range为P95−P05，低真值range不报ratio。固定面板N<5不认证空间图，支持统计单列不合格人口。
首次、连续3个1秒帧的稳定起点、最后确认帧分开；缺帧打断，尾部未确认右删失，首帧达标左删失。
未达标/未知保留，CDF用全部合格固定事件，不只对成功者平均；双方均可确定的时间差注明条件人口，并列其他结局。

shadow完整A={−1.2,0}、p={.1,.3,.5,.7,.9}、lead门={0,3,5,10}s，首次完整MDN概率越过即锁存。
lead必须>0且>=tau；0提前量迟报。deadline在观测窗口外且未报为Unknown，不把未来没导出部分算漏报。
共同受影响真实站点集合固定；no_input与丢文件区别。H为空比例NA；未知给失败率上下信息界，非95%CI。
并列事件内空间误报率/警报精确率及原概率评分；不声称每日连续假警报率。

ideal-cutoff、measured numeric map和measured compute+shared render FIFO场景分别保存；通信UNKNOWN。
实测计算包括全部station/grid概率就绪，不用均摊query时间推整图；IO/预处理/encoder/readout/概率/渲染/发布/存档分项。
每事件FIFO：前一帧服务未结束时下一帧排队，处理超过1秒记录积压。
render后的交付场景在pack用实际全量normal渲染时间补齐；这是离线共享比较图的场景，不冒充部署时延。
TTUM和shadow共用clocks.py，不从较低MAE直接推出更早警报或更小盲区。
