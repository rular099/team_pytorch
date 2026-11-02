#!/usr/bin/env python3
"""Preserve v1 and compose a 28-slide, meeting-to-meeting native revision."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

DECK=Path(__file__).resolve().parent
OLD=DECK.with_name('multistation_pga_exchange_20260930')
spec=importlib.util.spec_from_file_location('native_v1',OLD/'build_native_deck.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.DECK=DECK;b.FIG=DECK/'assets/figures';b.TAB=DECK/'assets/tables'
b.OUT=DECK/'multistation_pga_exchange_20260930_v2.pptx'
S=json.loads((b.TAB/'revision_summary.json').read_text());G=S['groups'];F=S['fields']
sl=None


def v(key,metric):return f'{G[key][metric]:.4f}'
def begin(title,source='上次交流 2026-07-26 → 本次交流 2026-09-30；新结果为 validation',section='交流进展'):
    global sl
    sl=b.new(title,source,section)
    return sl
def end(talk,cues='',source='assets/tables/revision_summary.json；TECHNICAL_DETAILS.md'):
    b.note(talk,cues,source)
    return len(b.slides)-1
def tr(model,pop,percent):return next(r for r in S['thresholds'] if r['model']==model and r['population']==pop and r['pga_percent_g']==percent)
def pair(r):return f"{r['point_f1_star']:.3f} / {r['point_pr_auc']:.3f}"


def replace_native(slide,old,new):
    """Replace text in runs, retaining native typography and table layout."""
    for obj in slide.shapes:
        frames=[obj.text_frame] if obj.has_text_frame else []
        if obj.has_table:frames.extend(c.text_frame for row in obj.table.rows for c in row.cells)
        for frame in frames:
            for paragraph in frame.paragraphs:
                for run in paragraph.runs:run.text=run.text.replace(old,new)


def main():
    # Reuse native components/unchanged source pages, writing only in the new directory.
    b.build()
    original=list(b.slides);replacements={}
    old_title=b.texts[14]
    b.texts[14]='Random 误差分层——少站、高 PGA 与高震级仍是主要难点'
    replace_native(original[14],old_title,b.texts[14])
    _,talk,cues,source=b.notes[14];b.notes[14]=(b.texts[14],talk,cues,source)
    replace_native(original[20],'normal；best ep32','normal；ep20低MAE / ep32略高R²')
    replace_native(original[25],
                   '复现：prepare_sample_scatter.py → prepare_report_assets.py → build_native_deck.py；来源与指标见 assets/tables/。',
                   '复现：prepare_revision_assets.py → build_revision_deck.py；历史与当前口径见 sources.md / TECHNICAL_DETAILS.md。')
    replace_native(original[25],'RT59 correction：reports/rt59_v3_review_correction_20260920/',
                   'RT55 normal ep20（metadata核验）；七月 rt48 原始预测重算。')

    begin('从上次交流到本次——normal 精度大幅改善，新增 random 挑战')
    cards=[('上次 full-data MAE','0.2966','699 train / 101 val events',b.GRAY),
           ('本次 normal MAE',v('rt55_normal_all','mae'),'RT55 ep20 validation，89,770 targets',b.TEAL),
           ('Normal remote MAE',v('rt55_normal_noninput','mae'),'同模型非输入目标，26,119 targets',b.BLUE),
           ('新增 random MAE',v('rt61_random_all','mae'),'RT61 ep8 validation，75,654 targets',b.CORAL)]
    for i,(label,value,detail,c) in enumerate(cards):b.metric_card(sl,label,value,detail,.45+i*3.85,1.35,3.55,c)
    b.card(sl,'本次最重要的应用进展','旧 full-data → normal 代表\nMAE：0.2966 → 0.1162（差 −0.1804）\nR²：0.5221 → 0.7524\n训练规模：699 → 9,627 events',.45,3.65,7.35,3.45,b.TEAL,22)
    b.card(sl,'新增任务与仍存在的问题','normal 是常规实时应用，不可省略\nrandom 是更难的随机观测/远程查询\n单站空间反差、强 PGA 仍难\nRT61 最新，但不是所有场景最优',8.05,3.65,7.5,3.45,b.CORAL,22)
    b.conclude(sl,'这是跨交流的历史变化；数据与协议也变了，不把差值归因于单个模块。')
    replacements[1]=end('先回答上次到这次是否进步。最强normal代表的整体MAE已降到0.1162，独立test另有证据。Random是新增的更难任务，不应把它较低的R²误写成常规场景退步。旧KNET rt48的0.2877也保留在后表，不回避上次最佳诊断候选。','RT55 ep20为本次已完成normal validation中最低MAE的代表，ep32的R²略高；不拼接不同模型最优单项。')

    begin('技术演进——每轮解决什么问题，而不只是实验名字',section='模型路线')
    items=[('上次交流','波形/多站模型已能拟合\n跨事件泛化仍不足',b.GRAY),('RT55 / RT56','全量 normal 基线\n适配随机观测几何',b.BLUE),('RT57','让不同台站波形\n在表示中可区分',b.TEAL),('RT58','预测输入站强度锚点\n向目标位置迁移',b.BLUE),('RT59','同址局部 / 远程迁移\n分路学习与保留',b.TEAL),('RT60 / RT61','读出层强调空间反差\n增加波形×几何交互',b.CORAL)]
    for i,(title,body,c) in enumerate(items):
        x=.45+i*2.55;b.shape(sl,x,1.70,2.36,2.65);b.text(sl,title,x+.16,1.96,2.03,.52,24,c,True);b.text(sl,body,x+.16,2.80,2.03,1.35,17)
        if i<5:b.arrow(sl,x+2.36,3.0,x+2.53,3.0,c)
    b.card(sl,'术语翻译','station-distinctive：台站差异表示\nanchor transfer：波形锚点迁移\nlocal / transport：同址 / 远程残差修正',.45,4.83,7.35,2.60,b.BLUE,21)
    b.card(sl,'完整细节另附，不占主报告篇幅','contrast：同一场景不同目标的强弱反差\nwave × geometry：波形状态与目标几何乘性交互\n详见 TECHNICAL_DETAILS.md 第 2–4 节',8.05,4.83,7.5,2.60,b.TEAL,20)
    b.conclude(sl,'RT60、RT61 都从 RT59 出发；保留 normal 主线，研究 random 与空间场。')
    replacements[5]=end('这页将名字翻译成研究问题。RT57解决站点表示相似，RT58尝试波形锚点迁移，RT59把同址与远程分开，RT60只更新最后读出，RT61再加入低秩条件交互。具体数学、输入、冻结边界和失败原因放在独立答疑文档。','RT60不是RT61的parent；不宣称五次续训都带来全面提升。')

    begin('总体指标——上次基线、本次 normal 与新增 random 同页呈现')
    entries=[('上次','full-data validation','old_full_all'),('上次','rt48 KNET diagnostic','old_knet_all'),
             ('本次 normal','RT55 ep20 · all','rt55_normal_all'),('本次 normal','RT55 ep20 · non-input','rt55_normal_noninput'),
             ('本次最新 normal','RT61 ep8 · all','rt61_normal_all'),('本次最新 normal','RT61 ep8 · non-input','rt61_normal_noninput'),
             ('本次新增 random','RT61 ep8 · non-input','rt61_random_all')]
    rows=[[label,model,f"{G[key]['targets']:,}"]+[v(key,m) for m in ['mae','rmse','r2','slope']] for label,model,key in entries]
    b.table(sl,['交流 / 场景','模型与目标总体','targets','MAE ↓','RMSE ↓','R² ↑','slope →1'],rows,.45,1.42,15.1,4.65,[2.3,4.0,1.6,1.6,1.6,1.5,1.6],17,highlight=(2,3))
    b.card(sl,'Normal 不能被 random 淹没','最低点误差代表：RT55 ep20\nep32 all：MAE 0.1166 / R² 0.7541\n同址与远程必须同时报告',.45,6.15,7.35,1.62,b.TEAL,16)
    b.card(sl,'并列展示，不混合排名','旧 full-data 与 KNET controlled 不是同一协议\nRT55 normal 与 RT61 random 不是同一难度\nMAE / RMSE 单位：log10(m/s²)',8.05,6.15,7.5,1.62,b.BLUE,16)
    b.conclude(sl,'常规应用整体精度已明显提升；远程目标的 R² 与空间反差仍不能由 all-target 高分代替。')
    replacements[9]=end('本页主线改为交流间比较。RT55保留的normal能力是重要应用成果，RT61展示最新研究能力。旧数据集和现在不同，random也不是旧normal的直接续跑，因此不在表格里做单一胜负排行。','normal ep32不是ep20；表中全部为validation，独立test另行标注。')

    begin('交流间的主要增量——更大规模、常规场景精度与任务范围')
    b.picture(sl,b.FIG/'meeting_point_progress.png',.45,1.3,15.1,4.6)
    b.card(sl,'本次 normal 相对旧 full-data','MAE 差 −0.1804；RMSE 差 −0.2063\nR² 增加 0.2303；slope 0.5931 → 0.7634',.45,6.10,7.35,1.60,b.TEAL,17)
    b.card(sl,'新能力与比较边界','25 年全部训练 shards；新增 arbitrary random geometry\n旧/新目标不同，不给配对 CI 或单模块贡献比例',8.05,6.10,7.5,1.60,b.BLUE,17)
    b.conclude(sl,'进展不只来自结构：样本规模、网络筛选、实时输入和训练目标也共同变化。')
    replacements[10]=end('柱图从上次交流起点出发，包含旧full-data和旧最佳KNET诊断。现在normal的误差低很多；random任务更难，因此其R²低于旧锚点并不否定normal进步。历史差值是描述性变化，不是严格配对的因果收益。')

    begin('旧版散点对照——normal 更贴近真实值，random 仍有明显压缩','旧 Slide11 原始 NPZ → 本次 RT55 ep20 normal / RT61 ep8 random；均为 validation')
    b.picture(sl,b.FIG/'meeting_scatter_comparison.png',.45,1.50,15.1,5.76)
    b.text(sl,'统一轴域 [−3.0, 1.1]，等比例坐标，全点保留；旧 α=0.30，新 α=0.08。数据与目标协议不同。',.68,7.42,14.65,.40,17,b.GRAY)
    b.conclude(sl,'左到右：上次系统锚点 → 本次 normal all → normal non-input → 新增 random。')
    replacements[11]=end('这页直接满足旧图与新图对照。Normal代表来自RT55 ep20，random来自最新RT61。Normal all明显更贴近1:1，远程目标与random仍压缩；旧图与当前不同样本，点云密度不直接比较数量。')

    # Original evidence page still gives all four latest RT61 populations.
    begin('Random geometry 分层——台站数与截止时刻（RT61）','RT61 ep8 seed42 random validation；全部目标均为 non-input')
    b.picture(sl,b.FIG/'rt61_stratified_scatter.png',.45,1.3,11.95,6.49)
    b.card(sl,'实际输入台站数','K=1 MAE 0.2790\nK=2–5 0.2300\nK≥6 0.2105\n单站 slope 0.1886',12.6,1.5,2.95,3.1,b.BLUE,18)
    b.card(sl,'实时截止时刻','early 0.2325\nmid 0.2433\nlate 0.2408\n目标集合也变',12.6,4.98,2.95,2.63,b.CORAL,18)
    b.conclude(sl,'这是 random 分层，不是 normal；时间趋势不能解释为固定场景的因果收益。')
    replacements[13]=end('明确纠正原第14页标识：全部是random geometry结果。输入台站从因果可用站中随机选择，目标排除输入站。这页与下一页使用同一个RT61 checkpoint，但目标和台站集合不同。')

    begin('Normal geometry 分层——多站与后期预测显著改善（RT61）','RT61 ep8 seed42 normal validation；图中 all-target，右侧另列 non-input')
    b.picture(sl,b.FIG/'normal_geometry_stratified_scatter.png',.45,1.3,11.95,6.49)
    b.card(sl,'All-target MAE','K=1 0.2526\nK=2–5 0.1724\nK≥6 0.1025\nlate R² 0.9250',12.6,1.5,2.95,3.1,b.TEAL,18)
    b.card(sl,'Remote 需单列','K≥6 MAE 0.1830\nlate MAE 0.1868\n远程进步较小\n不只看 all 高分',12.6,4.98,2.95,2.63,b.BLUE,17)
    b.conclude(sl,'All-target 的后期收益含直接观测目标增加；远程预测不等于已观测站重建。')
    normal_page=end('这里新增normal分层。随时间增加，all-target MAE从early约0.189到late约0.055，明显优于random趋势；但remote从0.201到0.187，改善小很多。必须解释目标组成变化，不能把后期高R²推广到远程任意位置。','early=1/3/5、mid=10/20、late=40/90秒；实际K由station_valid_count核验。','normal_geometry_strata.csv；TECHNICAL_DETAILS.md。')

    begin('空间强弱反差——normal 恢复较好，random 仍明显压缩','P95−P05 比值来自旧/新原始预测重算；同事件/同截止时刻为一个 field')
    b.picture(sl,b.FIG/'meeting_spatial_range.png',.45,1.58,10.6,4.42)
    b.card(sl,'“空间范围比”不是面积','预测各目标的 P95−P05\n÷真实各目标的 P95−P05\n=台站间 log PGA 强弱反差\n0：压平；1：相似幅度',11.25,1.50,4.3,3.0,b.BLUE,19)
    b.card(sl,'容易理解的例子','真实跨度 1.0 dex\n预测跨度 0.4 dex\n范围比 = 0.4\n仍须检查位置排序是否正确',11.25,4.74,4.3,2.66,b.TEAL,18)
    b.card(sl,'两次交流的空间强弱反差对照',f"旧 full-data：{F['old_full_all']['range_ratio_mean']:.3f}\n本次 normal all：{F['rt55_normal_all']['range_ratio_mean']:.3f}；non-input：{F['rt55_normal_noninput']['range_ratio_mean']:.3f}\n新增 random：{F['rt61_random_all']['range_ratio_mean']:.3f}",.45,6.10,10.6,1.65,b.BLUE,16)
    b.conclude(sl,'n≥5 有效目标，field 等权；同一公式不代表同一事件/目标协议，不能给纯架构提升结论。')
    replacements[15]=end('空间范围比是场内PGA反差而非地图面积。使用旧full-data原NPZ重算，与本次normal和random用同一公式比较。Normal总体约0.83，random约0.49，说明最强点预测并未解决随机场动态范围。范围比1也不保证空间形状正确。','图中normal代表为RT55 ep20；完整源表同时给RT61 normal all 0.8278、non-input 0.6800，以及单站结果。')

    begin('概率输出的交流间进展——明显缓解上次的过度自信')
    b.picture(sl,b.FIG/'meeting_calibration_progress.png',.45,1.40,11.0,4.8)
    b.card(sl,'上次真实 validation','rt48 KNET\n1σ coverage 11.3%\n2σ coverage 21.4%\nBrier 0.2305',11.7,1.43,3.85,2.80,b.GRAY,19)
    b.card(sl,'本次 normal 代表','RT55 ep20\n1σ 70.2% / 2σ 95.4%\nBrier 0.0911\n最新 normal / random 同列',11.7,4.58,3.85,2.98,b.TEAL,18)
    b.text(sl,'上次 Slide27 报告 rt44–51 均值 13.8% / 27.0%；这里取实际 rt48 导出作单模型对照，不混用均值。',.67,6.74,10.5,.78,17,b.GRAY)
    b.conclude(sl,'校准行为大幅改善；样本与协议仍不同，高斯覆盖率参考不等于混合分布精确可信区间。')
    replacements[16]=end('这页不再从RT57开始。旧PPT的概率问题是KNET八轮模型过度自信，选其rt48实际导出对照，本次normal代表与最新normal/random都展示。Brier统一既定−1.2logPGA阈值。覆盖率改善很明显，但非高斯MDN不宜仅凭两条高斯参考线宣布完全校准。','旧rt48为4,082 targets，当前normal为89,770，random为75,654，不能把coverage变化解释成配对百分点因果提升。')

    begin('文献与两次交流对照——补齐 normal，并复现旧评分规则','20s validation；本项目列为 point-score F1* / PR-AUC；TEAM 为完整日本 test EEW')
    rows=[]
    for percent,tf,ta in [(1,.73,.82),(2,.69,.76),(5,.63,.68)]:
        rows.append([f'{percent}%g',f'{tf:.3f} / {ta:.3f}',pair(tr('old_threshold','all',percent)),
                     pair(tr('rt55_normal','all',percent)),pair(tr('rt55_normal','noninput',percent)),pair(tr('rt61_random','all',percent))])
    b.table(sl,['PGA阈值','TEAM test','上次 rt48 val','本次 normal all','本次 normal remote','本次 random'],rows,.45,1.40,15.1,2.63,[1.2,2.5,2.7,2.8,3.0,2.6],17,highlight=())
    b.text(sl,'本项目 F1*：在各 validation 快照内优化 point-score cutoff，仅复现旧探索规则，不是部署阈值或独立 test。',.67,4.18,14.6,.70,18,b.CORAL,True)
    b.card(sl,'Normal all 已不是“明显差很多”','20s RT55 normal all：12,874 targets\nremote：1,186 targets，5%g仅16正例\n输入目标更容易；远程与random仍难\n不能与完整EEW/提前量评价直接排名',.45,5.21,7.35,2.45,b.TEAL,18)
    b.table(sl,['回归量级 / 条件','R²','σ residual ln'],[
        ['QF forecasting seen / unseen','≈0.917 / ≈0.837','0.57 / 0.75'],
        ['上次 full-data validation',v('old_full_all','r2'),f"{G['old_full_all']['residual_std_ln']:.3f}"],
        ['本次 RT55 normal all',v('rt55_normal_all','r2'),f"{G['rt55_normal_all']['residual_std_ln']:.3f}"],
        ['本次 RT61 random',v('rt61_random_all','r2'),f"{G['rt61_random_all']['residual_std_ln']:.3f}"]],8.05,5.21,7.5,2.45,[4.3,1.7,1.6],15)
    b.conclude(sl,'同评分规则可定位历史变化；任务、地区、目标组成、cutoff 与预警时间差异仍阻止直接排名。')
    replacements[17]=end('文献页补齐normal代表和remote。旧三行点评分F1和PR-AUC已经从原NPZ精确重算，现在按相同point score、相同20秒和最优validation cutoff规则显示。F1星号只是探索性上限，不能视为部署性能。右表补旧full-data和最强normal，而不再只放random。','TEAM论文表为概率预警完整时间；QF为catalog forecasting且station-heldout，R²原图近似；current non-input不等于unseen station。','https://arxiv.org/pdf/2009.06316v4；https://arxiv.org/html/2412.00815v1；meeting_threshold_comparison.csv。')

    begin('为什么上一版文献表显得变差——三种口径混在了视觉主线里')
    b.card(sl,'旧报告：train 和 val 并列','1%g 的 train：0.965 / 0.995\n真正 val：0.620 / 0.642\n接近或超过文献的高分是容量\n不能把 train 当验证泛化成绩',.45,1.48,4.86,4.50,b.GRAY,21)
    b.card(sl,'上一版：遗漏重要 normal','只强调 RT61 random 与 remote\n忽略本次最强 normal 应用结果\n本次 normal all F1* 0.918\nPR-AUC 0.959（20s point score）',5.57,1.48,4.86,4.50,b.TEAL,21)
    b.card(sl,'F1 的规则也发生了改变','旧：point score 的 val 最优 cutoff\n新：MDN probability cutoff=0.5\nrandom F1* 0.639 vs fixed 0.609\n这不是相同规则下的模型退步',10.69,1.48,4.86,4.50,b.BLUE,21)
    b.text(sl,'部署型固定概率结果仍保留：normal all 1%g F1 0.916 / PR-AUC 0.978；random 为 0.609 / 0.707。',.68,6.55,14.6,.80,20,bold=True)
    b.conclude(sl,'准确结论：normal 应用与校准进展明显；random 和远程强震仍有差距，尚无同协议文献复现。')
    why_page=end('用户记忆有其来源：旧图中训练高分很突出，旧F1阈值在validation内优化。上一版还把更难random放在正中央，弱化normal。因此不能把视觉上差很多解释成系统全面退步。现在既呈现旧规则对照，也保留固定MDN阈值。','没有用新阈值选择checkpoint或打开test；数据任务差异只作边界说明，不当作已经证明的全部原因。')

    # Select originals/replacements, inserting exactly two additional main pages.
    order=[]
    for i in range(26):
        order.append(replacements.get(i,i))
        if i==13:order.append(normal_page)
        if i==17:order.append(why_page)
    titles=[b.texts[i] for i in order];notes=[b.notes[i] for i in order]
    title,talk,cues,source=notes[0]
    talk='本次汇报以七月交流为起点，先展示常规 normal geometry 的系统进展，再介绍新增 random 任务、空间场瓶颈和下一步数据机制实验。研究过程与实现细节在附录及独立技术说明中。'
    notes[0]=(title,talk,cues.replace('主报告 20 页','主报告 22 页'),source)
    b.slides[0].notes_slide.notes_text_frame.text=talk+'\n\n注意点：\n'+notes[0][2]+'\n\n来源：'+source
    slides=[b.slides[i] for i in order]
    ids=list(b.prs.slides._sldIdLst)
    for i,element in enumerate(ids):
        b.prs.slides._sldIdLst.remove(element)
        if i not in order:b.prs.part.drop_rel(element.rId)
    for i in order:b.prs.slides._sldIdLst.append(ids[i])
    # Chart records refer to final positions, never creation order.
    records=[]
    for number,slide in enumerate(slides,1):
        for obj in slide.shapes:
            if obj.name.startswith('Evidence '):
                path=b.FIG/obj.name.removeprefix('Evidence ')
                from PIL import Image
                with Image.open(path) as im:w,h=im.size
                records.append(dict(slide=number,shape=obj.name,path=str(path.relative_to(DECK)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_width_px=w,source_height_px=h))
    assert len(slides)==28 and len(notes)==28
    b.prs.core_properties.subject='2026-09-30 修订版；22 页主报告 + 6 页附录'
    b.prs.save(b.OUT)
    (DECK/'speech.md').write_text('# 修订版逐页讲稿\n\n'+'\n\n'.join(f'## Slide {i}: {title}\n\n{talk}\n\n注意点：\n{cues}\n\n来源：{source}' for i,(title,talk,cues,source) in enumerate(notes,1))+'\n')
    manifest=dict(pptx=b.OUT.name,slides=28,main_slides=22,appendix_slides=6,slide_titles=titles,image_records=records,
                  font=b.FONT,base_commit='8f54477ed11bf2623f40121648f7fcc756ada1a2',
                  pptx_sha256=hashlib.sha256(b.OUT.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  backend='native editable; historical scientific PNGs embedded unchanged',v1_preserved=True,
                  normal_representative='RT55 ep20 validation',normal_stratified='RT61 ep8 validation; all-target plus remote numeric companion',
                  random_representative='RT61 ep8 validation',historical_baseline='July full-data + rt48 KNET / threshold diagnostics',
                  literature_point_f1_star_is_validation_oracle_not_deployment=True)
    (DECK/'native_deck_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pptx':str(b.OUT),'slides':28,'main':22,'appendix':6,'scientific_figures':len(records),'v1_preserved':True},ensure_ascii=False))


if __name__=='__main__':main()
