#!/usr/bin/env python3
"""Native, editable 26-slide technical report; scientific images embedded intact."""
import csv
import hashlib
import json
import math
import shutil
import os
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

DECK=Path(__file__).resolve().parent
ROOT=DECK.parents[2]
REPORT=ROOT/'reports/rt61_wave_geometry_validation_20260928'
FIG=DECK/'assets/figures'
TAB=DECK/'assets/tables'
OUT=DECK/'multistation_pga_exchange_20260930.pptx'
FONT='Noto Sans CJK SC'
FONT_PATH=os.environ.get('NATIVE_REPORT_FONT', str(Path.home() / '.fonts/noto/NotoSansCJKsc-Regular.otf'))
BG='FAFBFC'; INK='2F3742'; BLUE='5B7FA3'; TEAL='55A6A0'; CORAL='D66F63'; BORDER='CCD8E2'; PALE='F2F6F8'; WHITE='FFFFFF'; GRAY='677683'
prs=Presentation(); prs.slide_width=Inches(16); prs.slide_height=Inches(9)
prs.core_properties.title='多台站波形驱动的任意目标位置 PGA 估计'
prs.core_properties.subject='2026-09-30 内部技术交流；20 页主报告 + 6 页附录'
prs.core_properties.author='TEAM PyTorch 项目组'
prs.core_properties.keywords='RT61,RT59,PGA,validation,random geometry,native editable PPT'
slides=[]; image_records=[]; texts=[]; notes=[]
groups=list(csv.DictReader((REPORT/'group_metrics.csv').open()))
M={(v['population'],v['model']):v for v in groups}
summary=json.loads((REPORT/'summary.json').read_text())
thresholds=list(csv.DictReader((TAB/'rt61_threshold_metrics.csv').open()))
residuals={v['population']:v for v in csv.DictReader((TAB/'rt61_literature_residual_metrics.csv').open())}


def rgb(value):return RGBColor.from_string(value)
def inch(value):return Inches(value)
def shape(slide,x,y,w,h,fill=WHITE,line=BORDER,rounded=True):
    q=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,inch(x),inch(y),inch(w),inch(h))
    q.fill.solid();q.fill.fore_color.rgb=rgb(fill)
    if line:q.line.color.rgb=rgb(line);q.line.width=Pt(.7)
    else:q.line.fill.background()
    if rounded:q.adjustments[0]=.045
    # Override theme effects so evidence panels remain flat, without inherited shadows.
    q._element.spPr.append(OxmlElement('a:effectLst'))
    return q


def text(slide,value,x,y,w,h,size=21,color=INK,bold=False,align=PP_ALIGN.LEFT):
    q=slide.shapes.add_textbox(inch(x),inch(y),inch(w),inch(h))
    q.name=f'Text {len(slide.shapes)}'
    tf=q.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(str(value).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line;p.alignment=align;p.space_after=Pt(size*.08);p.line_spacing=1.08
        for run in p.runs:
            run.font.name=FONT;run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=rgb(color)
    return q


def arrow(slide,x1,y1,x2,y2,color=BLUE):
    from pptx.oxml.xmlchemy import OxmlElement
    q=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,inch(x1),inch(y1),inch(x2),inch(y2))
    q.line.color.rgb=rgb(color);q.line.width=Pt(1.8)
    tail=OxmlElement('a:tailEnd');tail.set('type','triangle');q.line._get_or_add_ln().append(tail)
    return q


def wrapped_lines(value,width,size):
    """Conservative native-text height estimate in points, including explicit blank lines."""
    font=ImageFont.truetype(FONT_PATH,max(1,round(size*4)))
    limit=width*72*.94
    result=0
    for paragraph in str(value).split('\n'):
        line='';count=1
        for char in paragraph:
            if line and font.getlength(line+char)/4>limit:
                count+=1;line=char
            else:line+=char
        result+=count
    return result


def card(slide,title,body,x,y,w,h,color=BLUE,size=21):
    shape(slide,x,y,w,h)
    shape(slide,x+.06,y+.12,.055,h-.24,fill=color,line=None,rounded=False)
    title_h=max(.38,wrapped_lines(title,w-.5,23)*23*1.12/72+.035)
    body_offset=.18+title_h+.12
    body_h=h-body_offset-.14
    paragraphs=len(str(body).split('\n'))
    body_size=size
    while body_size>13:
        required=(wrapped_lines(body,w-.5,body_size)*body_size*1.12+paragraphs*body_size*.08)/72
        if required<=body_h:break
        body_size-=.5
    text(slide,title,x+.25,y+.18,w-.5,title_h,23,color,True)
    text(slide,body,x+.25,y+body_offset,w-.5,body_h,body_size)


def metric_card(slide,label,value,detail,x,y,w,color=BLUE):
    shape(slide,x,y,w,1.85)
    text(slide,label,x+.2,y+.17,w-.4,.4,18,GRAY)
    text(slide,value,x+.2,y+.62,w-.4,.67,36,color,True)
    text(slide,detail,x+.2,y+1.38,w-.4,.34,13,GRAY)


def picture(slide,path,x,y,w,h):
    path=Path(path)
    with Image.open(path) as im: iw,ih=im.size
    scale=min(w/iw,h/ih);pw,ph=iw*scale,ih*scale
    q=slide.shapes.add_picture(str(path),inch(x+(w-pw)/2),inch(y+(h-ph)/2),width=inch(pw),height=inch(ph))
    q.name='Evidence '+path.name
    image_records.append({'slide':len(slides),'shape':q.name,'path':str(path.relative_to(DECK)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_width_px':iw,'source_height_px':ih,
        'shape_width_emu':q.width,'shape_height_emu':q.height,'crop':[q.crop_left,q.crop_right,q.crop_top,q.crop_bottom]})
    return q


def table(slide,headers,rows,x,y,w,h,widths=None,size=17,highlight=()):
    t=slide.shapes.add_table(len(rows)+1,len(headers),inch(x),inch(y),inch(w),inch(h)).table
    if widths:
        for column,width in zip(t.columns,widths):column.width=inch(w*width/sum(widths))
    for i,row in enumerate([headers]+rows):
        for j,value in enumerate(row):
            c=t.cell(i,j);c.text=str(value)
            c.margin_left=inch(.12);c.margin_right=inch(.08);c.margin_top=inch(.065);c.margin_bottom=inch(.04)
            c.vertical_anchor=MSO_ANCHOR.MIDDLE
            c.fill.solid();c.fill.fore_color.rgb=rgb(BLUE if i==0 else ('E7F1EF' if i-1 in highlight else (WHITE if i%2 else PALE)))
            for p in c.text_frame.paragraphs:
                p.alignment=PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER
                p.space_after=Pt(0);p.line_spacing=1.0
                for run in p.runs:
                    run.font.name=FONT;run.font.size=Pt(size);run.font.bold=i==0;run.font.color.rgb=rgb(WHITE if i==0 else INK)
    return t


def new(title,source='RT61 formal validation · epoch 8 · seed 42',section=''):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);slides.append(sl);texts.append(title)
    sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(BG)
    text(sl,title,.45,.27,15.1,.64,29 if len(title)<34 else 26,bold=True)
    shape(sl,.45,1.00,15.1,.028,fill=BLUE,line=None,rounded=False)
    if section:text(sl,section,12.0,.88,3.4,.21,9,GRAY,align=PP_ALIGN.RIGHT)
    text(sl,source,.46,8.66,15.04,.25,9,GRAY)
    return sl


def conclude(sl,value,color=BLUE,y=7.93):
    shape(sl,.45,y,15.1,.58,fill=PALE)
    text(sl,value,.67,y+.1,14.65,.39,20,color,True)


def note(talk,cues,source):
    content=talk+'\n\n注意点：\n'+cues+'\n\n来源：'+source
    slides[-1].notes_slide.notes_text_frame.text=content
    notes.append((texts[-1],talk,cues,source))


def four(v):return f'{float(v):.4f}'
def value(pop,key,model='candidate_rt61'):return four(M[pop,model][key])
def threshold(pop,percent,seconds=20):
    return next(r for r in thresholds if r['population']==pop and int(r['pga_percent_g'])==percent and int(r['requested_time_seconds'])==seconds)


def build():
    # 01: two text objects only, exactly as requested.
    sl=prs.slides.add_slide(prs.slide_layouts[6]);slides.append(sl);texts.append('多台站波形驱动的任意目标位置 PGA 估计')
    sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(WHITE)
    text(sl,'多台站波形驱动的\n任意目标位置 PGA 估计',1.0,3.0,14.0,1.7,38,bold=True,align=PP_ALIGN.CENTER)
    text(sl,'2026-09-30',1.0,5.4,14.0,.5,21,GRAY,align=PP_ALIGN.CENTER)
    note('今天汇报随机观测几何下 PGA 预测的最新进展。重点看点误差、未观测目标、空间场和概率校准，以及这些结果是否足以支持下一轮模型决策。','内部技术交流；主报告 20 页，附录 6 页。','报告日期 2026-09-30；源代码分析基点 8f54477。')

    sl=new('结论先行——点预测改善，空间场目标尚未完成')
    for i,(label,v,detail,c) in enumerate([
        ('Random MAE',value('random','mae'),'75,654 个非输入目标',BLUE),
        ('Random R²',value('random','r2'),'回归斜率仍为 0.4113',BLUE),
        ('Normal all MAE',value('normal_all','mae'),'含 63,651 个输入站目标',TEAL),
        ('Normal non-input MAE',value('normal_noninput','mae'),'26,119 个远程目标',TEAL)]):metric_card(sl,label,v,detail,.45+i*3.85,1.35,3.55,c)
    card(sl,'同口径点预测收益','相对 RT57 γ=1.66\nRandom MAE 降低 3.36%\nNormal non-input MAE 降低 5.29%',.45,3.65,7.35,3.45,TEAL,22)
    card(sl,'是否晋级是另一项判断','RT61 单站 pairwise MAE 仅改善 0.385%\n未达预设 2% 实用进步门槛\n保留 RT59 为 development parent',8.05,3.65,7.5,3.45,CORAL,22)
    conclude(sl,'RT61 是最新小幅改进候选；不是已完成空间泛化目标的新主线。')
    note('先给结论。RT61 的随机几何点误差略有改善，相对较早的 RT57 参考已有稳定收益；但相对 RT59 的增量很小，不能只凭最小 MAE 决定更换主线。空间差分仍未达到预注册门槛，所以当前保留 RT59。下面将把四类目标分开，避免把输入站的高精度当作远程推断能力。','全页为 validation，单 seed 42。RT57 为同次导出的 γ=1.66 frozen historical reference，不是原始未缩放 RT57。','group_metrics.csv；decision.json；useful_joint_progress_gates.csv。')

    sl=new('任务升级——随机观测台站，预测随机目标位置','任务协议来自 sampler / mask / forward 定义',section='任务与数据')
    card(sl,'Normal geometry','沿用实时近站/先触发输入协议\n目标包含输入站与非输入站\n同时检验原任务保留能力',.45,1.38,7.25,2.72,BLUE,22)
    card(sl,'Random geometry','只从截至 t 已触发、波形有效的站选输入\n随机台站集合、随机目标几何\n正式 random 目标排除输入台站',8.0,1.38,7.55,2.72,TEAL,21)
    stages=[('截至时刻 t 的三分量波形','坐标 + storage-valid mask'),('可变数量台站集合','K 取实际有效输入数'),('目标位置查询','同址 local / 远程 transport'),('概率 PGA 输出','混合分布 + log10 均值')]
    for i,(a,b) in enumerate(stages):
        x=.65+i*3.84;shape(sl,x,4.85,3.5,1.5);text(sl,a,x+.15,5.05,3.2,.43,19,BLUE,True);text(sl,b,x+.15,5.73,3.2,.35,16)
        if i<3:arrow(sl,x+3.5,5.6,x+3.77,5.6)
    text(sl,'任务不是“波形与台站错配”：每段波形始终与真实输入台站身份、坐标一致。',.7,6.83,14.6,.6,22,CORAL,True)
    conclude(sl,'非输入目标是核心应用对象；已观测同址目标是保留能力，不应混为同一种难度。')
    note('这里的随机几何不是把波形乱配到另一台站，而是在因果可用输入中随机选择台站。波形和真实台站坐标始终正确对应。Normal 允许目标与输入同址，random 正式评估排除输入站，因此二者的总体 MAE不能直接互相排名。远程位置预测是更关键的能力。','不要将 non-input 自动称为 unseen station：当前没有独立的 station-heldout split。','gemini_util_light.py；RT56/RT59/RT61 validation config；正式 observable route 导出。')

    sl=new('相关工作——与 TEAM 最接近，与 QuakeFormer 任务可相互启发','TEAM: arXiv:2009.06316v4；QuakeFormer: arXiv:2412.00815v1',section='相关工作')
    for x,title,body,c in [(.45,'GMPE / ASK14','震源参数 + 距离 + 场地\n→ 震动强度分布\n\n强项：物理量可解释\n边界：依赖震源估计，波形信息有限',GRAY),
        (5.57,'TEAM','实时多台站波形 + 坐标\n→ 任意位置概率 PGA\n\n最接近本项目的 EEW 范式\n需同时评价预警准确率与提前量',BLUE),
        (10.69,'QuakeFormer','空间 / 波形 / 时间 mask\n→ forecasting、interpolation、EEW\n\n统一表征与任务迁移\nseen / unseen station 单列评估',TEAL)]:card(sl,title,body,x,1.35,4.86,4.70,c,19)
    shape(sl,.45,6.40,15.1,1.13,fill=PALE)
    text(sl,'本项目：冻结 DiTing 波形骨干；研究随机几何下的站点辨识、local / remote 路由和空间差异恢复。',.75,6.69,14.5,.6,22,bold=True)
    conclude(sl,'借鉴任务设计与评价维度；跨地区、跨任务的文献数字只作参照，不作为排行榜。')
    note('TEAM 是最直接的任务参照；QuakeFormer 的价值是统一遮蔽任务和显式空间依赖。我们当前仍在冻结波形骨干的约束下尝试小规模残差机制。因此后面既报告预测误差，也报告阈值预警、空间差分和未观测站点指标，不能仅用一个 R² 概括系统。','原文核验：TEAM supplement Table S1；QuakeFormer Fig. 3 / Table 3。图示为原生概念摘要，不冒充论文原图。','https://arxiv.org/pdf/2009.06316v4；https://arxiv.org/html/2412.00815v1；ASK14 由 QuakeFormer 原文引用。')

    sl=new('训练使用 25 年全量 shards；当前新结果仍全部是 validation','Japan 2000–2024 · event-exclusive split within annual shards',section='数据与协议')
    for i,(label,v,d) in enumerate([('TRAIN events','9,627','146,842 station rows'),('VALIDATION events','1,383','22,336 station rows'),('HELD-OUT TEST events','2,769','42,368 station rows；未用于 RT57–61')]):metric_card(sl,label,v,d,.45+i*5.13,1.30,4.84,TEAL if i==1 else BLUE)
    table(sl,['协议','实时行数','有有效目标事件','有效目标数','时刻'],[
        ['Normal','9,681','1,383','89,770','1 / 3 / 5 / 10 / 20 / 40 / 90 s'],
        ['Random','9,681','1,310','75,654','同上；事件数按导出 event ID 去重']],.45,3.60,15.1,1.66,[1.6,1.5,2.,1.5,5.],17)
    card(sl,'训练范围与固定预算','RT61 继承全部 25 个年度训练 shards\n从 RT59 epoch 8 微调 8 个新 epoch\n固定 final epoch 8，seed 42；不扫参数、不挑 checkpoint',.45,5.57,7.35,1.99,BLUE,17)
    card(sl,'证据边界','年度内事件互斥，不是 held-out-year 外推\nnormal 与 random 共用固定验证事件清单\n反复用于开发的 validation 不替代独立 test',8.05,5.57,7.5,1.99,CORAL,17)
    conclude(sl,'“全量训练数据”成立；“新模型已通过独立测试”目前不成立。')
    note('训练不是 512-event 子集。RT61 的 resolved contract 已确认 2000到2024全部25个年度 shards。但新的正式预测只来自固定验证集。Random 的1310个事件指有有效目标的事件，不应误写为重新分出的更小验证集。目标数是事件、实时截止时刻和目标站记录之和，不能当独立样本数。','全量指本项目过滤后的训练划分，不是所有原始日本记录无过滤；test 绝不参与本轮选择。','summary.json training_config_contract / counts；PROJECT_CONTEXT split snapshot。')

    sl=new('技术演进——从容量诊断到点预测与空间场的联合验收','2026-07 → 2026-09；全程保留 RT55 加载 / 推理兼容路径',section='技术路线')
    items=[('2026-07','699-event 锚点\n512-event 容量诊断',GRAY),('RT55 / RT56','normal / random 基线\nRT55 ep20 test',BLUE),('RT57','station-distinctive\n时序 / 幅度状态',BLUE),('RT58','anchor transfer field\n未形成有效替代',CORAL),('RT59','local / transport 双路\n当前开发主线',TEAL),('RT60 / RT61','contrast / wave-geometry\n部分收益，未晋级',CORAL)]
    for i,(label,body,c) in enumerate(items):
        x=.45+i*2.55;shape(sl,x,2.05,2.36,2.6);text(sl,label,x+.18,2.30,2.0,.5,24,c,True);text(sl,body,x+.18,3.12,2.,1.3,17)
        if i<5:arrow(sl,x+2.36,3.28,x+2.53,3.28,c)
    card(sl,'评价从一个均值扩展为四类证据','点预测：MAE / RMSE / R²\n空间场：range ratio / pairwise difference',.45,5.30,7.3,2.17,BLUE,21)
    card(sl,'不把机制负结果当作重复训练的理由','概率与保留：NLL / Brier / coverage\n先核验机制，再决定主线和独立 test',8.0,5.30,7.55,2.17,TEAL,21)
    conclude(sl,'目前主要收益来自更有区分度的站点表示与双路由；空间反差仍是关键瓶颈。')
    note('七月主要验证模型能否拟合，九月则转向任意观测几何和未观测目标。RT59 保留为主线，不是因为所有指标达标，而是它在既定边界下提供最有价值的点预测改进。RT60、RT61 是有明确机制问题的小实验，未晋级不代表毫无价值，也不应被表述为已经完成目标。','本页为机制路线，不用不同轮次的非配对数值画连续增长曲线。','RT57–RT61 CODEX_RESULT；各轮 RESULT_REVIEW；SESSION_SUMMARY。')

    sl=new('当前系统架构——波形站点状态、目标查询与残差双路由','模型实现：gemini_models.py；RT57 / RT59 / RT61 设计文档',section='模型架构')
    blocks=[('三分量波形','截至 t 的真实样本\n坐标 + valid mask'),('DiTing encoder','pre-pooling temporal tokens\n冻结预训练骨干'),('站点差异表示','temporal / amplitude / duration\nabsolute u + centered d'),('多台站与目标查询','TEAM-style context\nquery / station geometry'),('Frozen base MDN','目标 PGA 混合分布\nlogits / means / sigma')]
    for i,(title,body) in enumerate(blocks):
        x=.45+i*3.07;shape(sl,x,1.67,2.84,2.20,fill=PALE);text(sl,title,x+.15,1.9,2.54,.62,21,BLUE,True);text(sl,body,x+.15,2.69,2.54,1.0,16)
        if i<4:arrow(sl,x+2.84,2.82,x+3.04,2.82)
    text(sl,'RT57 temporal branch：直接使用 u、d 与目标几何形成 station-conditioned correction。',.65,4.14,14.7,.56,20)
    card(sl,'RT59：唯一同址 local / 远程 transport','输入坐标批次复用 station / event memory\nforward 只依赖坐标与有效 mask\n以目标专属修正平移全部 mixture means',.45,5.03,7.35,2.38,TEAL,20)
    card(sl,'RT61：wave × geometry residual adapter','只训练 rank-16 adapter + remote residual readout\n其余 RT59 state 冻结；normal input 增量为零\nlogits / sigma 保留，输出 mean 改变',8.05,5.03,7.5,2.38,CORAL,19)
    conclude(sl,'架构显式区分“站点波形信息”与“目标几何信息”；小残差实验仍不足以解决空间场。')
    note('上排给出共享系统，DiTing 的时序 token、幅度与有效时长共同形成站点表示，而不是只看一个高度相似的池化向量。目标查询结合台站与事件记忆产生基准 MDN。RT59 增加同址和远程双分支；RT61只在远程 pair representation 上检验低秩波形—几何交互。对RT61而言，上游都被冻结。','冻结状态针对 RT61 本轮，不等于所有历史训练阶段都冻结 TEAM。absolute u 与 residual d 同时使用，单站 d=0 不代表没有波形。','docs/rt57_gtnp_v2_station_distinctive.md；docs/rt59_dual_objective_transport_v3.md；docs/rt61_wave_geometry_residual.md。')

    sl=new('RT59 双路由——RT61 只检验远程波形 × 几何增量',section='模型机制')
    shape(sl,4.35,1.35,7.3,.95,fill=PALE);text(sl,'Query 与有效输入站是否唯一坐标匹配？',4.58,1.57,6.85,.45,23,BLUE,True)
    arrow(sl,5.1,2.3,3.95,3.0);arrow(sl,10.6,2.3,12.0,3.0)
    text(sl,'唯一匹配',2.63,2.63,2.5,.35,17,TEAL);text(sl,'无匹配 / 歧义',11.4,2.63,3.2,.35,17,CORAL)
    card(sl,'Local correction','同址输入站的局部残差修正\nRT61 保持 RT59 local 输出\nnormal input increment = 0',.45,3.00,7.35,2.40,TEAL,21)
    card(sl,'Transport correction','station memory × query geometry\nrank-16 wave × geometry residual\n只改变 remote correction',8.05,3.00,7.5,2.40,BLUE,21)
    shape(sl,.45,5.90,15.1,1.44,fill=PALE)
    text(sl,'RT61 增量：波形投影 × 几何投影 → 低秩交互 → zero-init 输出投影 → 新 readout',.7,6.10,14.6,.55,22,bold=True)
    text(sl,'Frozen RT59 teacher 与 RT61 student 同次 forward：权重 / distance gate 用原表示，避免参考漂移。',.7,6.84,14.6,.38,18)
    conclude(sl,'路由不用 PGA 标签、target type 或协议位；非输入不等于训练时未见台站。')
    note('Local 的唯一匹配完全由 forward 可见坐标和mask决定，重复坐标的歧义情况落到 transport。RT61 的输出投影零初始化，teacher始终使用原RT59表示，因此可以精确比较同一次forward的候选和参考。重要的是，只训练新adapter和remote readout，不能把它的结果归因于重新训练全模型。','共9个tensor / 99,621 scalars；forward 不读标签。normal input变化上限检查单独核验。','RT61 trainable_manifest.json；reference_invariance.json；模型 forward。')

    sl=new('训练目标——点误差、空间反差、概率与保留能力联合验收',section='训练与门槛')
    table(sl,['RT59 目标组','固定权重','作用'],[['Random non-input','0.50','学习任意几何远程预测'],['Normal input','0.25','保留观测同址能力'],['Normal other','0.25','保留常规远程能力']],.45,1.4,7.5,2.27,[2.5,1.,3.5],18)
    card(sl,'RT61 固定目标','grouped point + 0.05 × parent regret\n                         + 0.40 × field contrast\nNLL + 0.10 SmoothL1 + 0.10 MSE',8.2,1.4,7.35,2.27,BLUE,19)
    categories=[('Pointwise','MAE / RMSE / R² / 1 s 结果',BLUE),('Spatial field','slope / range ratio / pairwise Δ',TEAL),('Calibration','NLL / Brier / 1σ、2σ coverage',BLUE),('Retention','normal input / non-input 配对 CI',CORAL)]
    for i,(a,b,c) in enumerate(categories):card(sl,a,b,.45+i*3.85,4.35,3.55,2.45,c,18)
    text(sl,'RT61：14 个 mechanism gates + 25 个 required legacy gates + 4 个 useful-progress gates',.7,7.15,14.6,.45,20,bold=True)
    conclude(sl,'只有同时满足机制、保留能力和实用效应门槛，才考虑更换 development parent。')
    note('分组加权避免数量很多的normal input目标压过random和remote目标。验收要求是合取关系，不能把18/25当作总体72分就自动晋级。RT61有13个机制门槛通过，但normal non-input RMSE仍轻微退化；实用进步还要求单站pairwise改善至少2%。','ASCII: L = grouped_point + 0.05 * parent_regret + 0.40 * field_contrast.\nRaw LaTeX: \\mathcal L=\\mathcal L_{point}+0.05\\mathcal L_{regret}+0.40\\mathcal L_{contrast}.','RT59 / RT61 design docs；mechanism_gates.csv；legacy_gates.csv。')

    sl=new('最新总体指标——random 小幅最优，normal 保留但尚未晋级',section='总体结果')
    rows=[]
    for pop,label in [('random','Random non-input'),('normal_all','Normal all'),('normal_noninput','Normal non-input')]:
        for model,name in [('historical_rt57','RT57 γ=1.66'),('reference_rt59','RT59 reference'),('candidate_rt61','RT61 candidate')]:
            rows.append([label,name]+[value(pop,k,model) for k in ['mae','rmse','r2','slope']])
    table(sl,['目标总体','模型','MAE ↓','RMSE ↓','R² ↑','slope →1'],rows,.45,1.38,11.20,5.76,[2.6,2.7,1.5,1.5,1.5,1.6],16,highlight=(2,5,8))
    card(sl,'Random 概率分数',f"NLL  {value('random','nll')}\nBrier  {value('random','brier')}",11.9,1.38,3.65,2.45,BLUE,22)
    card(sl,'验收结论','mechanism  13 / 14\nlegacy  18 / 25\nuseful progress  0 / 4',11.9,4.1,3.65,3.02,CORAL,21)
    text(sl,'单位：MAE / RMSE 为 log10(m/s²)；RT57 γ、RT59、RT61 由同一正式 NPZ 配对导出。',.66,7.35,14.75,.32,14,GRAY)
    conclude(sl,'RT61 random MAE 优于 RT59；normal non-input RMSE 未优于 RT59，不能称“全面最好”。')
    note('表格突出的是最新候选行，不是对所有指标都标成最佳。Random MAE和RMSE都有小幅收益；normal all差异很小，normal non-input RMSE由0.25565变成0.25572，略有退化。这样的变化和置信区间共同说明，当前不能宣布一个全面优于父模型的新方案。','所有目标数固定；表中RT57使用γ=1.66 historical export。','group_metrics.csv；decision.json；legacy / mechanism / useful_joint_progress_gates.csv。')

    sl=new('严格同口径提升——RT57 γ=1.66 → RT59 → RT61',section='配对比较')
    picture(sl,FIG/'paired_mae_progress.png',.45,1.40,15.1,4.5)
    table(sl,['RT61 − RT59','点估计 ΔMAE','event-cluster 95% CI','判断'],[
        ['Random','−0.000505','[−0.000883, −0.000146]','小幅改善，CI 排除 0'],
        ['Normal non-input','−0.000058','[−0.000489, +0.000390]','很小且 CI 包含 0']],.7,6.15,14.6,1.28,[2.4,2.2,4.5,4.0],16)
    conclude(sl,'对 RT57 的收益稳定；对 RT59 的新增收益很小，远程 normal 保留没有显著优势。')
    note('同口径指同一组事件、时间和目标的paired comparison。相对RT57的random MAE降低3.36%，normal non-input降低5.29%；相对RT59则只降低约0.21%和0.03%。CI按事件聚类重采样，避免把同一事件的多个时刻与台站当成独立样本。','曲线纵轴为局部范围，标注精确数值；没有把跨模型bootstrap当作multi-seed不确定性。5000 draws, seed 20260915。','group_metrics.csv；paired_ci.csv；paired_mae_progress.png。')

    sl=new('旧版散点图对照——点误差更低，但远程目标仍有压缩','旧版 Slide 11 原始 NPZ 重绘；两版均为 validation；不作跨协议提升百分比',section='历史对照')
    text(sl,'2026-07：699 train / 101 val events，固定旧协议     →     当前：9,627 train / 1,383 val events，实时 mixed / random geometry',.63,1.29,14.75,.68,19,bold=True)
    picture(sl,FIG/'old_vs_rt61_scatter.png',.45,2.05,15.1,5.35)
    text(sl,'三图统一轴域 [−3.0, 1.1]、等比例坐标、保留全部点。旧图 α=0.30，新图 α=0.08，密度不可直接比数量。',.65,7.47,14.75,.36,14,GRAY)
    conclude(sl,'历史背景：规模与点误差改善；remote R² / slope 仍未超越旧锚点，不宣称严格全面提升。')
    note('用户提到的旧散点确实存在。这里从原始NPZ重绘，确认1393个有效目标和原报告指标完全一致。当前normal all明显更接近1:1线，但它包含大量输入台站目标；只看normal non-input则仍有明显压缩，R²也低于旧版。数据和目标协议不同，不能把这页的差异直接解释为模型架构带来的提升。','旧图best checkpoint epoch未在本次独立核验；只使用旧评估导出身份。旧版原图保持在历史目录，本页不是从生成式幻灯片抠图。','old_vs_rt61_metrics.csv；report_asset_provenance.json；July Slide 11 prompt / main_pga_metrics.csv；旧eval_results_best.npz。')

    sl=new('真实 PGA 与预测 PGA——输入站更准，远程目标仍有压缩',section='散点主证据')
    picture(sl,FIG/'rt61_scatter_four_populations.png',.45,1.20,9.15,6.66)
    card(sl,'输入站预测最准确','MAE  0.0945\nR²  0.8405\n已观测波形提供直接约束',9.85,1.36,5.70,1.94,BLUE,20)
    card(sl,'远程预测更困难','Normal non-input MAE  0.1992\nRandom geometry MAE  0.2373\n应用评价需单列非输入目标',9.85,3.57,5.70,1.99,TEAL,19)
    card(sl,'动态范围仍被压缩','Random slope  0.4113\nNon-input slope  0.5154\n高 PGA 目标仍有低估',9.85,5.84,5.70,1.99,CORAL,20)
    conclude(sl,'全量有效点，α=0.08；虚线为 1:1，珊瑚实线为 OLS，等高线为 50% / 80% / 95% 密度范围。',y=8.06)
    note('先看normal input，它最接近对角线。再看normal non-input和random，拟合线更平，强震端容易被拉回事件平均水平。这里的点估计是在log10坐标上的混合均值，不是取最强分量，也不是在线性PGA空间求期望。原始图直接嵌入，无裁切、无拉伸，所有有效点均参与统计。','原始PNG 3072×2520，PPT中只等比缩放。点数为target-time记录，不是独立事件。','rt61_sample_panel_metrics.csv；rt61_sample_figure_provenance.json；正式random / normal epoch8 NPZ。')

    sl=new('散点分层——多站改善明显，时间收益不是单调关系',section='分层结果')
    picture(sl,FIG/'rt61_stratified_scatter.png',.45,1.30,11.95,6.49)
    card(sl,'实际台站数','K=1 MAE  0.2790\nK=2–5  0.2300\nK≥6  0.2105\n\n单站 slope 仅 0.1886',12.6,1.45,2.95,3.35,BLUE,18)
    card(sl,'输入时间','early  0.2325\nmid  0.2433\nlate  0.2408\n\n集合与几何也在变化',12.6,5.08,2.95,2.60,CORAL,17)
    conclude(sl,'分层是描述性比较：不同 K / 时刻的目标集合不同，不能据此推断增加时长的因果收益。')
    note('上排按真实有效K分组，单站的整体斜率只有0.1886，说明它最容易接近常数输出。下排按时间分组，early的MAE反而更低，但后期R²提高，这不矛盾，因为不同组的真实PGA分布和目标集合变化。Random每个时刻重新选择输入，不能把这页当作严格同一场景时间扫参。','每组全量有效点，相同轴域和透明度；early=1/3/5，mid=10/20，late=40/90秒。','rt61_scatter_strata.csv；正式 NPZ station_valid_count / requested_elapsed_time。')

    sl=new('误差分层——少站、高 PGA 与高震级仍是主要难点',section='分层结果')
    picture(sl,FIG/'rt61_error_strata.png',.45,1.30,11.85,6.45)
    card(sl,'高 PGA 风险','truth ≥−0.5\nMAE  0.5510\n5,026 targets\n\n分桶 bias 见源表',12.6,1.55,2.95,2.65,CORAL,19)
    card(sl,'高震级长尾','M≥7：仅 6 events\n624 target-time记录\nMAE  0.5023\n\n不是 624 个独立样本',12.6,4.53,2.95,2.96,BLUE,18)
    conclude(sl,'尚未触发目标比已触发非输入更难；长尾结果需同时报告事件数与目标数。')
    note('只看总体平均会掩盖长尾。随机几何中高PGA分桶的MAE约0.551，明显高于总体0.237。M≥7只有6个事件，同一事件多时刻多台站形成624个记录，不能据此给出很窄的泛化保证。台站数趋势也受到事件难度和几何覆盖影响，不能简单认定越多台站必然越好。','分层为random正式目标；不把按幅值分桶的偏差解释成单独的训练因果机制。','rt61_error_strata.csv；正式 strata_metrics.csv。')

    sl=new('空间场指标——小幅恢复反差，仍未达到预注册目标',section='空间场')
    picture(sl,FIG/'field_gate_progress.png',.45,1.40,15.1,4.48)
    card(sl,'样本与定义','仅 n≥5 有效 query 的 field\nRandom 5,762 fields；K=1 有 1,494 fields\nP95−P05 空间范围比，field 等权',.45,6.04,7.35,1.54,BLUE,16)
    card(sl,'RT61 对 RT59 的实际增量','单站 pairwise MAE：0.3362 → 0.3349\n改善 0.385%，低于实用门槛 2%\n空间范围比仍为 0.1905 < 0.25',8.05,6.04,7.5,1.54,CORAL,16)
    conclude(sl,'点误差降低没有充分转化为正确的场内差异；单输入场景仍只恢复少量真实空间范围。')
    note('这里在每个事件—时间场内比较不同query，所以能够回答系统是否真的生成空间差异。RT61相对RT59恢复了一些范围，但相对RT57历史参考仍更压缩。单站pairwise MAE的改善虽然CI排除0，效应只有0.385%，未达到预设2%，不宜把统计显著写成实用突破。','空间范围采用P95−P05，不是max−min；field等权，至少5个有效目标。完整固定案例可在原始fixed_examples.png审阅。','summary.json random_fields；field_metrics.csv；paired_ci.csv；fixed_examples.csv。')

    sl=new('概率输出与校准——概率分数改善，coverage 保真仍不够',section='概率评估')
    for i,(label,v,detail,c) in enumerate([('Random NLL','0.1962','RT57 γ: 0.2175；越低越好',BLUE),('Random Brier','0.1834','RT57 γ: 0.1899；越低越好',TEAL),('Normal all NLL','−0.8482','连续密度 NLL 可为负',BLUE),('Normal all Brier','0.0953','正式既定 PGA 阈值',TEAL)]):metric_card(sl,label,v,detail,.45+i*3.85,1.27,3.55,c)
    picture(sl,FIG/'coverage_comparison.png',.45,3.48,10.1,4.2)
    card(sl,'不能由一个概率分数\n宣称完全校准','1σ coverage 相对 RT57 变化：\nrandom +2.14 pp\nnormal all +1.51 pp\n均超过 ±1 pp 保真门槛\n\nRT59–61 logits / sigma 冻结',10.8,3.50,4.75,4.14,CORAL,18)
    conclude(sl,'coverage 是均值±预测 σ 的经验覆盖率；高斯参考线不等于混合分布精确可信区间。')
    note('NLL和Brier相对RT57都改善，说明均值修正有概率收益。但这不是联合优化不确定性的结果，RT59到RT61基本不改变logits和sigma。coverage变化超过保真门槛说明输出分布的行为发生变化，不等于它一定朝坏方向变化。混合分布可能非高斯，所以图中的高斯覆盖率只是参考。','Brier使用正式评估既定阈值，与后续1/2/5%g探索表分开。pp是百分点。','group_metrics.csv；legacy_gates.csv；reference_invariance.json。')

    sl=new('文献指标对照——用于定位差距，不构成直接排名','TEAM Table S1；QuakeFormer Fig. 3 / Table 3；RT61 固定 20 s MDN exceedance',section='文献比较')
    text(sl,'TEAM 风格阈值：F1 / PR-AUC',.65,1.26,7.55,.45,23,BLUE,True)
    rows=[]
    for percent,f1,auc in [(1,.73,.82),(2,.69,.76),(5,.63,.68)]:
        r=threshold('random',percent)
        rows.append([f'{percent}%g',f'{f1:.3f} / {auc:.3f}',f"{float(r['f1_at_alpha_0p5']):.3f} / {float(r['pr_auc_trapezoid']):.3f}",r['positive_targets']])
    table(sl,['阈值','TEAM 日本 test','RT61 random val','RT61正例'],rows,.45,1.96,8.0,2.49,[1.,2.8,2.8,1.5],17)
    text(sl,'RT61：20 s，9,603 targets / 1,310 events，α=0.5 固定。\nTEAM：完整实时预警评价，α逐阈值优化；两者协议不同。',.66,4.65,7.7,.94,17)
    text(sl,'QuakeFormer：R² / 残差 σ（ln 单位）',8.77,1.26,6.70,.45,22,TEAL,True)
    table(sl,['条件','R²','σ residual ln'],[
        ['QF forecasting seen test','≈0.917','0.57'],['QF forecasting unseen test','≈0.837','0.75'],
        ['RT61 random validation','0.3994',f"{float(residuals['random']['residual_std_ln']):.3f}"],
        ['RT61 normal non-input val','0.4223',f"{float(residuals['normal_noninput']['residual_std_ln']):.3f}"]],8.78,1.96,6.77,3.1,[3.4,1.1,1.65],15)
    text(sl,'QF 为 California、目录源参数 forecasting；非实时波形同任务。\n本项目 non-input ≠ QF unseen station。R² 为原图近似读数。',8.98,5.26,6.32,.90,16)
    shape(sl,.45,6.45,15.1,1.08,fill=PALE)
    text(sl,'本项目已有 test 锚点：RT55 ep20 normal all MAE 0.1151 / R² 0.7578；non-input MAE 0.1947。',.7,6.66,14.5,.38,19,bold=True)
    text(sl,'这是独立模型与原协议的结果，不是 RT61 test，也不能与上表合并排名。',.7,7.14,14.5,.30,15,GRAY)
    conclude(sl,'Random 在强震阈值和远程空间辨识上仍有明显差距；normal all 高分不能替代 remote 证据。')
    note('TEAM表给出了相同PGA阈值的文献数字，RT61则使用MDN超阈值概率和固定0.5概率cutoff。我们没有在验证结果上优化F1，因此不与TEAM的F1最优cutoff作胜负结论。QuakeFormer一侧是catalog forecasting而不是相同的实时波形任务；残差std已换为ln单位，但地区、PGA定义和样本方差仍不同。RT55 ep20 test单列作历史锚点。','1%g / 2%g / 5%g用g=9.80665m/s²。σ为去bias后的残差std，不是RMSE。TEAM PR-AUC为表中AUC结合PR曲线的解释。','https://arxiv.org/pdf/2009.06316v4 (Table S1)；https://arxiv.org/html/2412.00815v1 (Fig.3, Table3)；rt61_threshold_metrics.csv；rt61_literature_residual_metrics.csv；RT55 aggregate CSV。')

    sl=new('当前问题——空间压缩、单站、长尾与独立泛化证据',section='局限与风险')
    table(sl,['问题','已核验证据','影响 / 边界'],[
        ['随机几何动态范围压缩','slope 0.4113 < 0.45','高 PGA 输出向事件均值收缩'],
        ['单站空间差异不足','range ratio 0.1905 < 0.25','难以恢复真实场内对比'],
        ['强震与高震级长尾','truth ≥−0.5 MAE 0.5510；M≥7仅6事件','风险不能由总体均值概括'],
        ['不确定性更新受限','RT59–61 mean-only correction','sigma/logits冻结；需单列coverage'],
        ['normal 新增收益不稳定','non-input ΔMAE CI 包含0','RT61未证明可靠超越RT59'],
        ['泛化证据不足','RT57–61固定validation、仅seed42','未证实station-heldout / 多seed / test']],.45,1.40,15.1,5.85,[3.1,5.6,5.5],19)
    conclude(sl,'已证实的是上述结果边界；“波形骨干导致瓶颈”仍需受控数据 / 表征实验，不是既定事实。',CORAL)
    note('问题需要分事实和假设。我们已经能看到空间场压缩和长尾困难，但不能仅凭这些现象断言DiTing骨干一定不适合PGA。冻结骨干、补零、幅度域和仪器响应都有可能影响结果，需要受控实验隔离。开发验证集已被反复使用，最终机制通过后仍需要独立泛化证据。','不将任意non-input位置称为真正训练未见station；不给出尚未运行的RT61 test数字。','formal metrics / gates / decision；SESSION_SUMMARY。')

    sl=new('阶段结论——保留 RT59，下一步先完成 V01 受控数据实验',section='结论与行动')
    card(sl,'已完成','全量 Japan 2000–2024 训练链\n更难的 random geometry 任务\n点预测与远程误差持续改善\n来源、mask与配对统计可追溯',.45,1.40,4.86,3.98,TEAL,21)
    card(sl,'尚未完成','单站场内差异和空间动态范围\nstrong-PGA / 高震级稳定性\n联合不确定性与校准\n新主线独立test / 多seed证据',5.57,1.40,4.86,3.98,CORAL,21)
    card(sl,'当前模型决策','RT61：部分机制有价值\n实用门槛未通过，不晋级\nRT59：继续作为开发 parent\nV01：独立数据机制实验',10.69,1.40,4.86,3.98,BLUE,21)
    text(sl,'近期：V01 三臂 vfull / vmissing / apair → 同事件 / 坐标 / cutoff 配对 → 决定新的表征学习方向',.7,5.93,14.5,.86,22,bold=True)
    text(sl,'V01 当前仅代码与数据准备完成，尚无确认成功的 HPC 结果；不能把它写成性能进展。',.7,7.05,14.6,.50,19,CORAL)
    conclude(sl,'先回答数据与输入机制问题；避免继续无明确机制假设的小幅残差微调。')
    note('总结不是继续重复当前设置。RT61已经回答了低秩波形—几何增量的可用性，但效应不足以换主线。V01将速度输入和P前缺失拆成受控问题，保持事件、坐标和时刻一致。现阶段报告只说明准备状态，不能声称V01已经带来收益。新的候选机制通过后，再安排held-out test而不是用test继续开发。','V01从预注册RT55 ep32起点，不是从RT61开始。速度灵敏度换算不等于完整去响应。','decision.json；SESSION_SUMMARY；docs/v01_velocity_prep_padding.md。')

    # Appendix, page ids remain 21–26 but no visible page-number decoration.
    sl=new('附录｜RT55–RT61 的实验身份与模型决策','历史报告与 formal result provenance；不同原始协议不混合排名',section='附录')
    table(sl,['实验','Parent / 起点','训练边界','主要观察','决策'],[
        ['RT55','Japan full baseline','normal；best ep32','ep20 test可核验；ep32 test未核验','兼容基线'],
        ['RT56','RT55 ep32','mixed normal/random；6 ep','random适配改善；normal遗忘','random基线'],
        ['RT57','RT56','station-distinctive head；6 ep','原始random MAE 0.2479；γ参考0.2456','保留机制'],
        ['RT58','RT57','anchor transfer；8 ep','anchor未改善；单站范围下降','不替代'],
        ['RT59','RT58 ep8','local/transport；8 ep','γ参考上的random MAE 0.2378','保留parent'],
        ['RT60','RT59 ep8','final contrast readout；8 ep','微小空间收益；normal remote退化','不晋级'],
        ['RT61','RT59 ep8','rank16 adapter + readout；8 ep','random 0.2373；useful 0/4','不晋级']],.45,1.45,15.1,5.80,[1.1,2.2,3.7,5.0,2.0],17)
    conclude(sl,'RT59 与 RT61 的具体量级比较优先使用 RT61 导出的 immutable same-forward reference。')
    note('这页用于复核实验身份。RT57原始epoch6结果和后续gamma=1.66冻结参考不是同一个数值口径，因此都注明。RT60和RT61同从RT59固定parent出发，并非RT61继承RT60。RT55 test目录名曾误导checkpoint身份，本报告只认metadata核验的epoch20。','目录名不是checkpoint身份；不同轮次不以小数尾差作为新发现。','RT55/56 PROJECT_CONTEXT；RT57–61 CODEX_RESULT；RT61 source identity。')

    sl=new('附录｜RT61 完整指标、formal counts 与概率输出','所有点误差为 log10(m/s²)；point estimate = mixture mean in this coordinate',section='附录')
    pops=['random','normal_all','normal_input','normal_noninput']
    rows=[['有效 events','1,310','1,383','1,383','1,290'],['有效 realtime rows'],['有效 targets']+[f"{int(M[p,'candidate_rt61']['targets']):,}" for p in pops]]
    # Rows with any query are authenticated from derived valid masks.
    counts=json.loads((TAB/'rt61_sample_figure_provenance.json').read_text())
    countmap={r['population']:r for r in counts['panels']}
    rows[1]=['有效 realtime rows']+[f"{countmap[p]['realtime_rows']:,}" for p in pops]
    for key,label in [('mae','MAE'),('rmse','RMSE'),('bias','Bias (pred−truth)'),('r2','R²'),('slope','OLS slope'),('p90_abs_error','P90 |error|'),('p95_abs_error','P95 |error|'),('within_0p1','|error|≤0.1 比例'),('within_0p2','|error|≤0.2 比例'),('nll','NLL'),('brier','Brier (formal阈值)'),('coverage1','Mean±1σ coverage'),('coverage2','Mean±2σ coverage'),('predictive_sigma_mean','Mean predictive σ')]:rows.append([label]+[value(p,key) for p in pops])
    table(sl,['指标 / count','Random','Normal all','Normal input','Normal non-input'],rows,.45,1.38,15.1,6.47,[3.5,2.7,2.7,2.7,3.1],14)
    conclude(sl,'导出总行数两协议均为 9,681；“有效行数”仅计至少一个有效目标的行，不应混淆。')
    note('完整表用于回答口径问题。点误差是target-weighted，空间指标则另用field等权。预测sigma是混合分布的二阶矩标准差。coverage使用均值正负sigma覆盖，不是按分位数计算的精确interval。有效行数按至少一个有效目标统计；唯一事件数按导出event ID去重，不能从总行数除以七反推。','全部metric来自正式CSV；events/有效rows来自正式NPZ的valid target mask。','group_metrics.csv；rt61_sample_figure_provenance.json。')

    sl=new('附录｜TEAM 风格阈值指标——固定 cutoff 与时间快照','g=9.80665 m/s²；MDN概率；α=0.5固定，无验证集阈值优化',section='附录')
    rows=[]
    for pop,label in [('random','Random'),('normal_noninput','Normal non-input'),('normal_all','Normal all')]:
        for percent in (1,2,5):
            r=threshold(pop,percent)
            rows.append([label,'20 s',f'{percent}%g',f"{int(r['targets']):,}",r['positive_targets'],f"{float(r['f1_at_alpha_0p5']):.3f}",f"{float(r['pr_auc_trapezoid']):.3f}"])
    table(sl,['目标总体','时刻','PGA阈值','targets','正例','F1@0.5','PR-AUC'],rows,.45,1.38,15.1,4.93,[3.4,1.,1.4,1.8,1.5,1.7,1.7],17)
    text(sl,'Normal non-input 的 5%g 只有 16 个正例 / 250 events；normal all 含大量直接观测站，不能代表远程预警。',.68,6.59,14.5,.66,19,CORAL,True)
    text(sl,'源表还保留 1 s、5 s 结果与 Average Precision；AP 未冒充梯形 PR-AUC；不计算提前量。',.68,7.39,14.5,.34,15,GRAY)
    conclude(sl,'TEAM 是完整实时报警结果，本页是 validation 时刻快照；只比较数量级与局限。')
    note('阈值先从%g转换到m/s²再取log10，随后对每个MDN分量求tail probability并按mixture weight求和。固定0.5判定计算F1；PR曲线对所有唯一概率分数分组，避免ties影响。图中normal non-input高阈值正例很少。源CSV还提供其他时刻用于透明审阅，没有用这些指标选择模型。','Terminal: p_exceed = sum_m weight[m] * P_m(PGA >= threshold).\nRaw LaTeX: p=\\sum_m w_m\\Pr_m(Y\\geq y_{th}).','rt61_threshold_metrics.csv；report_asset_provenance.json；TEAM Table S1。')

    sl=new('附录｜RT61 参数边界——99,621 scalars，其余冻结',section='附录')
    table(sl,['可训练部分','tensors','scalars','初始化 / 用途'],[['rank-16 adapter','3','32,544','wave / geometry projection；output projection zero-init'],['remote residual readout','6','67,077','在RT59 readout基础上更新'],['合计','9','99,621','白名单检查；不训练local / encoder / TEAM']],.45,1.45,15.1,2.50,[3.6,1.4,1.8,7.0],20,highlight=(2,))
    card(sl,'同次 reference 不漂移','共享state fingerprint核验\nimmutable RT59 readout hash核验\n候选与teacher配对导出\nnormal input increment最大值=0',.45,4.53,7.35,2.93,TEAL,21)
    card(sl,'RT55 兼容与本报告边界','新机制由显式flag开启\nRT55旧checkpoint加载 / 推理保持\n本轮仅新增报告脚本与素材\n未改训练、采样器或模型运行代码',8.05,4.53,7.5,2.93,BLUE,21)
    conclude(sl,'Mean-only correction：logits 与 component sigma 只存在数值精度级差异，不是新不确定性学习。')
    note('参数量准确来自运行时manifest，而不是手算层尺寸。额外小delta checkpoint仍依赖精确RT59 parent和共享状态hash，不能单独加载成完整模型。本报告制作没有改变任何训练或推理入口，因此不会影响现有超算训练和RT55兼容路径。','不宣称本报告额外重跑了全部模型回归；沿用已有兼容性验证记录。','trainable_manifest.json；reference_invariance.json；RT61 design doc；本报告文件变更清单。')

    sl=new('附录｜统计、门槛与复现边界','paired event-ID clustered bootstrap · 5,000 draws · seed 20260915',section='附录')
    card(sl,'统计单位','同一event的所有时刻和query一起重采样\ntarget-weighted point metrics\nfield等权，n≥5用于空间门槛\n单seed CI不代表训练seed不确定性',.45,1.4,7.35,3.07,BLUE,21)
    card(sl,'验收关系','14个mechanism gates：13通过\n25个required legacy gates：18通过\n4个diagnostic gates：不决定GO\n4个useful-progress gates：0通过',8.05,1.4,7.5,3.07,CORAL,21)
    table(sl,['关键门槛','观测值','预设条件','结论'],[['random slope','0.4113','≥0.45','未过'],['random range ratio','0.4855','≥0.55','未过'],['K=1 range ratio','0.1905','≥0.25','未过'],['K=1 pairwise MAE','0.3349','≤0.32948（对parent改善2%）','未过']],.45,4.94,15.1,2.5,[3.5,2.1,6.,2.],17)
    conclude(sl,'validation 被反复用于开发；下一阶段的独立 test 必须在候选与协议锁定之后。')
    note('门槛不是由本次画图重新制定，而来自预注册设计。特别是2%单站pairwise门槛不同于legacy绝对0.33门槛，两者在文件中都有独立记录。本报告不新增置信区间、不改变原有决策，只用源CSV和bootstrap结果解释边界。','CI来自既有正式报告；没有新增bootstrap选择或test阈值优化。','paired_ci.csv；mechanism_gates.csv；legacy_gates.csv；useful_joint_progress_gates.csv。')

    sl=new('附录｜参考文献与结果来源','文献固定版本、模型身份、科学图与来源哈希均记录在交付目录',section='附录')
    refs=[('TEAM','Münchmeyer, Bindi, Leser & Tilmann. The transformer earthquake alerting model.\narXiv:2009.06316v4；Table S1（日本test），完整实时预警。','https://arxiv.org/pdf/2009.06316v4'),
        ('QuakeFormer','Feng, Zhu & Lu. A Uniform Approach to Earthquake Ground Motion Prediction.\narXiv:2412.00815v1；Fig.3 / Table3，forecasting 与station-heldout。','https://arxiv.org/html/2412.00815v1'),
        ('数据来源','NIED K-NET, KiK-net（2019）. DOI: 10.17598/NIED.0004。\n本项目正式模型训练使用过滤后的Japan 2000–2024 K-NET-only split。','https://www.kyoshin.bosai.go.jp/en//about_pubdata/'),
        ('内部证据','RT61：reports/rt61_wave_geometry_validation_20260928/\nRT59 correction：reports/rt59_v3_review_correction_20260920/\n历史锚点：20260726 Slide11引用的原始eval_results_best.npz。','')]
    for i,(label,body,url) in enumerate(refs):
        y=1.37+i*1.47;shape(sl,.45,y,15.1,1.30);q=text(sl,label,.70,y+.16,2.3,.44,21,BLUE,True);text(sl,body,3.0,y+.16,12.2,1.05,16)
        if url:
            for p in q.text_frame.paragraphs:
                for r in p.runs:r.hyperlink.address=url
    text(sl,'复现：prepare_sample_scatter.py → prepare_report_assets.py → build_native_deck.py；来源与指标见 assets/tables/。',.7,7.43,14.6,.35,16,GRAY)
    conclude(sl,'分析仓库 rular099/team_pytorch · exp/v01-velocity-prep-padding-control · base 8f54477ed11b')
    note('交付目录包含科学图的PNG、PDF和SVG，以及派生指标、原始来源hash、构建脚本和逐页讲稿。PPT的文字、卡片、表格和架构都是原生对象；科学图作为原图等比例嵌入。旧图和新图的所有关键数字均可回到原NPZ或正式CSV核验。','本报告未提交新超算任务、未打开heldout test、未推送代码。公开文献引用固定v4/v1以避免版本漂移。','report_asset_provenance.json；native_deck_manifest.json；sources.md；speech.md。')

    assert len(slides)==26 and len(notes)==26
    prs.save(OUT)
    (DECK/'speech.md').write_text('# 技术交流讲稿\n\n'+ '\n\n'.join(
        f'## Slide {i}: {title}\n\n{talk}\n\n---\n\n注意点：\n{cues}\n\n来源：{source}'
        for i,(title,talk,cues,source) in enumerate(notes,1))+'\n')
    manifest={'backend':'native python-pptx; no generative full-slide images',
        'user_authorization':'2026-09-30 native editable PPT and old scatter comparison',
        'pptx':OUT.name,'slides':26,'main_slides':20,'appendix_slides':6,'font':FONT,
        'cover_rule':'title and date only; two text shapes',
        'chart_embedding':'original image bytes, uniform scaling, zero crop',
        'image_records':image_records,'slide_titles':texts,
        'base_commit':'8f54477ed11bf2623f40121648f7fcc756ada1a2',
        'pptx_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (DECK/'native_deck_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pptx':str(OUT),'slides':26,'notes':26,'scientific_images':len(image_records),
                      'layout':'native editable text, shapes and tables'},ensure_ascii=False))


if __name__=='__main__':build()
