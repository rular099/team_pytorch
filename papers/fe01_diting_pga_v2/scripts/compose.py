"""Generate readable tables, integrated HTML/Markdown and editable TeX without inference."""
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

import mistune
import pandas as pd

from common import *

def load(name):return pd.read_csv(PAPER/'assets/tables'/name)
def table(frame,columns,rename=None,digits=4):
    return md_table(frame[columns].rename(columns=rename or {}),digits)

t1=load('T1.csv');t2=load('T2_full.csv');t3=load('T3.csv')
t1text='## 表1 模型与实验设置\n\n'
t1text+=table(t1,['model','pretraining','encoder','native_samples','post_P_capacity_seconds','epoch_selected'],
    dict(model='模型',pretraining='预训练资产',encoder='骨干状态',native_samples='原生样本数',post_P_capacity_seconds='首P后窗能力 s',epoch_selected='selected轮次'),2)
t1text+='\n\n'
t1text+=table(t1,['model','total_parameters','trainable_parameters','encoder_parameters'],
    dict(model='模型',total_parameters='总参数',trainable_parameters='可训练参数',encoder_parameters='骨干参数'),0)
t1text+='\n\n四模型均训练12轮、2544更新，seed42；全局批量128、微批量8、Adam学习率0.001、余弦更新调度。前端投影和适配均可训练，共同读出为4层 target cross-attention 加3分量高斯MDN，物理幅值及有效时长共11维。PhaseNet取瓶颈掩码池化与线性投影；EQT取共享CNN/LSTM/Transformer主干后池化投影；DiTing使用归档ViT适配器；TEAM使用可训练CNN。参数量为归档接口审计记录，未在本轮加载权重计数。DiTing精确预训练语料/论文/许可尚未解决。原生窗能力不代表已测长时间性能。来源 metadata/checkpoint_inventory 与 evidence/complete_model_settings.json。\n'
(PAPER/'assets/tables/T1.md').write_text(t1text)

endpoint_names={'primary_10cell':'十单元主端点','first_second_2cell':'首秒两单元','early_6cell':'早期六单元'}
t2text='## 表2 主要预测结果\n\nvalidation，seed42，selected轮次按表1。误差坐标为log10(m/s²)，记为dex；除明确pooled栏外，单元内请求等权、单元间等权。RMSE为等单元MSE开方。Brier阈值−1.2；覆盖率为比例；NLL为负对数密度。为保证可读性，表2分点误差、概率评分和校准子表；每个端点与人口分别成块，不能跨块合并样本。\n'
for endpoint,title in endpoint_names.items():
    for population,popname in [('noninput','非输入目标'),('all','全部目标')]:
        part=t2[t2.endpoint.eq(endpoint)&t2.target_group.eq(population)].copy()
        part=part.set_index('family').loc[FAMILIES].reset_index()
        one=part.iloc[0]
        t2text+=f'\n### {title} {popname}\n\n政策 selected，轮次依次12/12/10/11；{int(one.events)}事件，{int(one.requests)}请求，{int(one.supported_cells)}单元。\n\n'
        for cols,ren in [
            (['family','equal_cell_mae','equal_cell_rmse','equal_cell_bias'],dict(family='模型',equal_cell_mae='MAE dex',equal_cell_rmse='RMSE dex',equal_cell_bias='bias dex')),
            (['family','equal_cell_nll','equal_cell_crps','equal_cell_brier'],dict(family='模型',equal_cell_nll='NLL',equal_cell_crps='CRPS dex',equal_cell_brier='Brier')),
            (['family','equal_cell_covered68','equal_cell_covered95','equal_cell_width95','equal_cell_predictive_sigma'],dict(family='模型',equal_cell_covered68='覆盖68',equal_cell_covered95='覆盖95',equal_cell_width95='宽度95 dex',equal_cell_predictive_sigma='sigma dex'))]:
            t2text+=table(part,cols,ren,6)+'\n\n'
t2text+='### 合并请求伴随统计\n\n以下仅列非输入主端点，其余在T2_pooled.csv。R²和斜率按合并请求计算，不是等单元统计。\n\n'
t2text+=table(t2[t2.endpoint.eq('primary_10cell')&t2.target_group.eq('noninput')],
    ['family','pooled_valid_targets','pooled_r2','pooled_prediction_on_truth_slope'],
    dict(family='模型',pooled_valid_targets='pooled请求',pooled_r2='pooled R²',pooled_prediction_on_truth_slope='预测对真值斜率'),6)
(PAPER/'assets/tables/T2.md').write_text(t2text+'\n')

t3text='## 表3 评价人口与支持范围\n\n'
t3text+='旧协议请求/新固定目标/空间场/预警受影响目标分别统计；TTUM与shadow资格不同。图5 unknown为事件数，图7 unknown为受影响目标数。下表不据未执行时钟组合填值。\n\n'
t3text+='未知数以DiTing为例，其他模型见S5/S6；人口分母为共同支持。\n\n'
t3=t3.replace({'protocol':{'legacy 5 times × 2 geometries':'旧五时点×两几何','fixed query normal':'固定目标 normal','fixed query random':'固定目标 random','shadow normal rendered −1.2 p.5 tau3':'失败 normal渲染 −1.2 p.5 tau3'},'population':{'all':'全部','noninput':'逐时点非输入','fixed_all_targets':'固定全部','fixed_never_input_targets':'固定never-input'},'unknown':{'not silently excluded; source metric status retained':'无失败请求'}})
t3['ineligible_events']=t3.ineligible_events.astype(object).where(t3.ineligible_events.notna(),'不适用')
t3['requests_or_fields']=t3.requests_or_fields.astype(object).where(t3.requests_or_fields.notna(),'固定目标见S5')
t3text+='shadow行的事件子集列指含受影响目标的事件，误报另用完整负例分母。\n\n'
t3text+=table(t3,['protocol','population','total_events','eligible_events','ineligible_events','requests_or_fields','unknown'],
    dict(protocol='协议',population='人口',total_events='原始事件',eligible_events='可评价事件',ineligible_events='不适格事件',requests_or_fields='请求/场/目标',unknown='未知'),0)
t3text+='\n\n旧协议为cutoff时钟；固定query normal有ideal/numeric/rendered，random只有ideal；shadow正文采用normal rendered。空间非输入944事件/6789场（pair为928/6644），全目标1057/9250（pair为1035/9037）。六事件计时每模型每scope108次，与全normal104800帧服务统计分开。缺失预测、资格不足和未知结局在各自原表保留，不删除后计算成功率。\n'
(PAPER/'assets/tables/T3.md').write_text(t3text)

captions=(PAPER/'figure_captions.md').read_text()
cap={m.group(1):m.group(2).strip() for m in re.finditer(r'## 图(\d+) ([^\n]+\n.*?)(?=\n## 图|\Z)',captions,re.S)}
def figure(i):
    body=cap[str(i)];title,body=body.split('\n',1)
    return f'\n\n![图{i} {title}](assets/figures/F{i}.svg)\n\n**图{i} {title}**  {body.strip()}\n'

ms=(PAPER/'manuscript.md').read_text()
full=ms.replace('## 2 数据与方法', '## 2 数据与方法')
for heading,addition in [
    ('### 2.3 模型设置',t3text),('### 2.4 实验设置',figure(1)),('### 2.5 评价指标',t1text),
    ('### 3.2 早期预测表现',t2text),('### 3.3 概率预测质量',figure(2)),
    ('### 3.4 空间预测能力',figure(3)),('### 3.5 回放与交付表现',figure(4)),
    ('### 3.6 分层与稳健性分析',figure(5)+figure(6)+figure(7))]:
    full=full.replace(heading,addition+'\n\n'+heading)
full=full.replace('完整核验条目及来源见 [references.md](references.md) 与 [references.bib](references.bib)。',
    (PAPER/'references.md').read_text().split('\n',1)[1])
(PAPER/'build/paper_full.md').write_text(full.rstrip()+'\n')

css='''
@font-face {font-family:PaperCJK;src:local("Noto Serif CJK SC");}
body{max-width:1040px;margin:40px auto;padding:0 32px;color:#202329;background:#fff;font-family:PaperCJK,"Noto Serif CJK SC",serif;font-size:17px;line-height:1.95;}
h1{text-align:center;font-size:30px;line-height:1.5;margin:1.8em 0;}h2{font-size:24px;border-bottom:1px solid #bbb;padding-top:1.2em;}h3{font-size:20px;margin-top:1.8em;}
p{text-align:justify;}img{max-width:100%;height:auto;display:block;margin:24px auto;}table{border-collapse:collapse;font:13px/1.6 "Noto Sans CJK SC",sans-serif;width:100%;margin:16px 0;}th,td{padding:7px 8px;border-bottom:1px solid #ccc;text-align:left;overflow-wrap:anywhere;}th{background:#eef3f8;border-top:2px solid #333;}a{color:#195b8a;}pre{font:13px/1.6 monospace;white-space:pre-wrap;background:#f5f6f7;padding:14px;}nav{font-family:sans-serif;font-size:14px;background:#edf2f7;padding:14px;}@media print{body{margin:0;padding:0;font-size:10.5pt;line-height:1.65;max-width:none;}p:has(>img){break-before:page;break-inside:avoid;break-after:avoid;margin:0;}p>img{max-height:183mm;max-width:100%;width:auto;object-fit:contain;break-inside:avoid;margin:0 auto;}p:has(>img)+p{font-size:8.5pt;line-height:1.4;break-inside:avoid;break-after:page;}table{font-size:8.5pt;break-inside:avoid;}pre{break-inside:avoid;}h2,h3{break-after:avoid;}nav{display:none;}@page{size:A4;margin:18mm;}}
'''
md=mistune.create_markdown(plugins=['table'])
def html_doc(title,text):return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>'+css+'</style><body>'+text+'</body></html>'
nav='<nav>审阅初稿 v2 · validation / seed42 · <a href="preview_supplement.html">补充材料</a> · <a href="preview_cases.html">全部案例</a> · <a href="GAPS.md">未决项</a> · <a href="HANDOFF_FOR_CHATGPT.md">交接</a></nav>'
(PAPER/'preview.html').write_text(html_doc('基于 DiTing 的早期 PGA 概率预测',nav+md(full)))
(PAPER/'preview_supplement.html').write_text(html_doc('补充材料',nav+md((PAPER/'supplement.md').read_text())))
idx=(PAPER/'assets/cases/CASE_INDEX.md').read_text();cases=md(idx.replace('](','](assets/cases/'))
case_rows=json.loads((PAPER/'assets/cases/manifest.json').read_text())
for i in range(0,len(case_rows),2):
    folder=Path(case_rows[i]['path']).parent.name
    cases+=f'<h2>案例 {i//2+1} {folder}</h2>'
    for row in case_rows[i:i+2]:cases+=f'<figure><img src="{row["path"]}"><figcaption>{html.escape(row["path"])}</figcaption></figure>'
(PAPER/'preview_cases.html').write_text(html_doc('全部原始预选案例',nav+cases))

# Editable TeX source; a successful build is separately reported, never assumed.
def texescape(text):
    for old,new in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('$',r'\$'),('#',r'\#'),('_',r'\_'),('{',r'\{'),('}',r'\}')]:
        if old=='\\':continue
        text=text.replace(old,new)
    return text
tex=[r'\documentclass[UTF8,a4paper,11pt]{ctexart}',r'\usepackage[margin=22mm]{geometry}',
     r'\usepackage{graphicx,hyperref,longtable,booktabs}',r'\begin{document}']
for line in ms.splitlines():
    if line.startswith('# '):tex.append(r'\title{'+texescape(line[2:])+r'}\maketitle')
    elif line.startswith('### '):tex.append(r'\subsection*{'+texescape(line[4:])+r'}')
    elif line.startswith('## '):tex.append(r'\section*{'+texescape(line[3:])+r'}')
    else:
        line=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',line)
        tex.append(texescape(line))
tex.append(r'\clearpage\section*{正文图}')
for i in range(1,8):tex.extend([r'\begin{figure}[p]\centering',rf'\includegraphics[width=\textwidth,height=.87\textheight,keepaspectratio]{{assets/figures/F{i}.pdf}}',r'\caption{'+texescape(cap[str(i)].replace('\n',' '))+r'}\end{figure}'])
tex.append(r'\clearpage\section*{表格}')
for text in [t1text,t2text,t3text]:
    in_table=False
    for line in text.splitlines():
        if line.startswith('|'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(c=='---' for c in cells):continue
            if not in_table:tex.append(r'\begin{longtable}{'+'l'*len(cells)+'}');in_table=True
            tex.append(' & '.join(texescape(c) for c in cells)+r' \\')
        else:
            if in_table:tex.append(r'\end{longtable}');in_table=False
            if line.startswith('#'):tex.append(r'\par\textbf{'+texescape(line.lstrip('# '))+r'}\par')
            else:tex.append(texescape(line))
    if in_table:tex.append(r'\end{longtable}')
tex.append(r'\end{document}')
(PAPER/'paper.tex').write_text('\n'.join(tex)+'\n')
checks=dict(html='GENERATED; visual inspection recorded separately',editable_tex='GENERATED',
    latex_engine=shutil.which('xelatex'),latex_pdf='NOT_BUILT; no TeX engine installed',
    figures_svg_pdf_png='GENERATED',main_table_groups=3,main_figure_groups=7)
write_json(PAPER/'qa/build_checks.json',checks)
print('PASS: integrated Markdown/HTML, 3 table groups, supplement/case previews, editable TeX; TeX PDF NOT_BUILT')
