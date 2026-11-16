"""Attach the permitted claim IDs to exact source rows and manuscript locations."""
import json
import ast
from common import *

e=Evidence()
frames={k:e.eid(k) for k in ['E03','E04','E05','E08','E09','E10','E11','E13','E14','E15','E17','E18']}
def s(k,**keys):
    d=frames[k]
    for key,value in keys.items():d=d[d[key].isin(value) if isinstance(value,list) else d[key].eq(value)]
    if not len(d):raise ValueError((k,keys))
    return dict(source_id=k,filter=keys,rows=json.loads(d.to_json(orient='records',double_precision=15)))
selected=dict(checkpoint_policy='selected')
rows={
 'C01':('3.1',['F2','T2'],[s('E03',**selected,target_group=['all','noninput'],endpoint='primary_10cell'),s('E04',**selected,target_group=['all','noninput'],endpoint='primary_10cell',metric=['mae','rmse'])]),
 'C02':('3.2',['F2','T2'],[s('E03',**selected,target_group='noninput',endpoint='first_second_2cell'),s('E04',**selected,target_group='noninput',endpoint='first_second_2cell'),s('E05',**selected,factor='time')]),
 'C03':('3.3',['F3','T2'],[s('E03',**selected,target_group='noninput',endpoint=['primary_10cell','first_second_2cell']),s('E04',**selected,target_group='noninput',endpoint=['primary_10cell','first_second_2cell'],metric=['nll','crps','brier'])]),
 'C04':('3.6',['S2'],[s('E04',**selected,target_group=['K1_noninput','Kmulti_noninput','untriggered_noninput'],endpoint='primary_10cell',metric='mae')]),
 'C05':('3.4',['F4'],[s('E08',**selected,target_group=['all','noninput'])]),
 'C07':('3.5',['F5'],[s('E10',budget_dex=.2,gate='field_pass',horizon=20),s('E11',budget_dex=.2,gate='field_pass')]),
 'C08':('3.5',['F5'],[s('E13',budget_dex=.2,gate='field_pass',geometry_protocol='normal',panel='fixed_never_input_targets',scene='ideal_cutoff_replay')]),
 'C09':('3.5',['F6'],[s('E14'),s('E15')]),
 'C10':('3.5',['F7'],[s('E17',threshold_logpga=-1.2,p_alarm=.5,tau=3),s('E18')]),
 'C11':('3.6',['S2'],[s('E05',**selected,factor='truth_logpga',stratum='y>0')]),
 'C12':('3.6',['S4'],[s('E09',**selected,target_group='noninput'),s('E04',checkpoint_policy='epoch12',target_group='noninput',endpoint='primary_10cell',metric='mae')])}
register=json.loads((PAPER/'evidence/blueprint_v2/04_CLAIM_REGISTER.json').read_text())
claims=[]
index=json.loads((PAPER/'evidence/blueprint_v2/05_EVIDENCE_INDEX.json').read_text())['evidence']
for item in register['claims']:
    cid=item['claim_id']
    if cid=='C06':continue
    section,assets,inputs=rows[cid]
    for inp in inputs:
        source=next(x for x in index if x['id']==inp['source_id'])
        record=next(x for x in e.records if x['path']==source['path'] and x['full_commit']==source['commit'])
        inp.update(full_commit=source['commit'],path=source['path'],sha256=record['actual_sha256'])
    claims.append(dict(claim_id=cid,statement=item['statement'],manuscript_section=section,assets=assets,
        status='SUPPORTED_WITH_STATED_LIMITATIONS',scope=item['scope'],limitations=item['limitations'],inputs=inputs))
write_json(PAPER/'evidence/claim_trace.json',dict(claims=claims,excluded_claim_ids=['C06'],
    note='source rows are evidence; blueprint rounded anchors used only for QA, never plotted'))
text='# 结论与来源映射\n\nvalidation / seed42；selected轮次D12/P12/E10/T11，epoch12只作敏感性。差值为DiTing减对照；CI为固定模型事件聚类探索性区间。每个定量结论的完整人口、请求数、单位与来源原行见 claim_trace.json。\n\n'
text+='|结论|正文|图表|来源|边界|\n|---|---|---|---|---|\n'
for row in claims:
    text+='|'+row['claim_id']+'|'+row['manuscript_section']+'|'+','.join(row['assets'])+'|'+','.join(x['source_id'] for x in row['inputs'])+'|'+row['limitations']+'|\n'
text+='\nC06不进入摘要、正文、图注、结论或补充材料。其他编号保持蓝图稳定身份，不重新编号为其他结论。\n'
(PAPER/'claim_map.md').write_text(text)

ranges=json.loads((PAPER/'evidence/source_function_ranges.json').read_text())
important={'fe01/data.py':['read_event','prepare_sample','geometry','TrainingDataset'],
 'fe01/windows.py':['clock','build_window'],'fe01/model.py':['shape_and_scale','FE01FullModel','_encode_station_waveform','build_model'],
 'fe01/engine.py':['evaluation_rows','loss','train','export_evaluation'],
 'fe01/metrics.py':['decode','score_rows'],
 'fe01/extractors.py':['PhaseNetBottleneck','EQTSharedTrunk','StationFrontend'],
 'fe01_a01/model.py':['build_model'],
 'gemini_models.py':['_station_memory_for_readout','forward','_pga_point_mean_from_output','MixtureOutput','CrossAttentionReadout'],
 'fe01_a01_compare/map_readiness.py':['frame_metrics','gate','times_to_ready'],
 'fe01_a01_compare/shadow.py':['probabilities','classify'],
 'fe01_a01_compare/clocks.py':['fifo'],'fe01_a01_compare/latency.py':['benchmark']}
doc='# 实际模型路径核验\n\n仓库 rular099/team_pytorch；论文分支 paper/fe01-diting-pga-manuscript-v2；论文基点 '+BASE+'；执行来源 '+MAIN+'；结果来源 '+RESULT+'。所有文件通过固定Git对象读取，未导入模型或加载权重。\n\n'
doc+='验证的路径为 dataset/sampler → 截止和存储掩码 → 物理波形与坐标 → 前端 → 目标读出 → MDN/损失 → 导出/回放。历史 FE01_PROTOCOL 只作名义协议，实际执行以完整配置和源分支为准。\n\n'
for path,names in important.items():
    doc+='## '+path+'\n\n'
    for row in ranges[path]:
        if row['name'] in names:
            url=f'https://github.com/rular099/team_pytorch/blob/{MAIN}/{path}#L{row["line"]}'
            doc+=f'- [{row["name"]}，{row["line"]}–{row["end_line"]}]({url})。\n'
doc+='''
## 已证实分支

四模型的 station_context_mode 为 off，pga_use_event_context 为 false，pga_readout_mode 为 target_cross_attention，pga_readout_layers 为4，readout_first_residual 为 true。绝对坐标启用；相对坐标、RoPE、Vs30、目标时间池化与 PGA 时间残差均未启用。十一维物理量与时长仍在实际输入中。图1不把未激活的台站自注意力接到PGA路径，也不把辅助震级/位置预测输入PGA。

shape_and_scale对有效掩码计算均值、尺度及有效时长，非有效样本不进入统计；前向输入没有最终标签。read_event只读取排他cutoff上界之前的波形，但允许读取最终PGA作监督；prepare_sample将输入、标签分开。冻结骨干在eval/no_grad边界内执行，投影/适配与读出可训练。

## 身份与表示边界

DiTing完整模型的历史配置与接口审计只用于确认本文候选身份；不取其他模型条件结果补齐论文。完整设置见 complete_model_settings.json，检查点轮次/校验值见固定结果 metadata/checkpoint_inventory.json。原始私有config摘要与去路径配置副本的字节摘要不同，本轮只核验可见表示与实际标志，未声称重新读取私有原件。精确预训练论文、语料与许可未查明。

## 评价与代价

station gate的阈值为MAE≤b、|bias|≤b/2、至少90%目标误差≤2b；field gate再约束centered RMSE和range_error≤b。三连续帧第三帧才确认，确认后已知不通过另计。shadow及时要求提前量严格为正且≥tau；未受影响的完整负例才进入FPR。FIFO逐事件逐模型计算，含渲染队列中的后帧数值已受前帧渲染阻塞。
'''
doc+='\n## 空间指标补充核验\n\n'
for path in ['fe01_review/diagnostics.py','fe01_a01_compare/spatial_statistics.py']:
    code=e.text(path,BASE,'spatial definition/weighting audit; not a claim of new execution')
    relevant=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name in ['field_metrics','field_intervals']]
    for n in relevant:
        doc+=f'- [{path}::{n.name} {n.lineno}–{n.end_lineno}](https://github.com/rular099/team_pytorch/blob/{BASE}/{path}#L{n.lineno})。\n'
doc+='\n此处核读论文基点的定义文件，不把它们混写为新回放执行版本。range为P95−P05；等距目标对采用震中距之差≤5 km的既定规则；level/shape/field使用均值误差平方、中心化误差平方均值、原误差平方均值。场bootstrap对支持单元等权，并保留空单元和有效重复数。\n'
(PAPER/'executed_model_path.md').write_text(doc)
e.save();print('PASS: 11 permitted claims with exact source rows and executed-path trace')
