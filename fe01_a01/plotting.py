"""Diagnostic figures and source CSVs, with explicit evidence labels."""
from pathlib import Path
import pandas as pd


def plot_diagnostics(root,evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);figures=root/'figures';figures.mkdir(exist_ok=True)
    stages=pd.read_csv(root/'query_stages.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    for k,group in stages.groupby('effective_keys'):
        table=group.groupby('stage',sort=False).query_rms.mean()
        ax.plot(range(len(table)),table,label=f'K={k}',marker='o')
        ax.set_xticks(range(len(table)),table.index,rotation=35,ha='right')
    ax.set_ylabel('Cross-query RMS (native units)');ax.set_yscale('symlog',linthresh=1e-8)
    ax.set_ylim(bottom=0)
    ax.set_title(evidence);ax.legend();fig.tight_layout();fig.savefig(figures/'query_flow.png',dpi=160);plt.close(fig)
    gates=pd.read_csv(root/'gates.csv')
    columns=[c for c in gates if 'gate' in c]
    fig,axes=plt.subplots(1,2,figsize=(12,4));ax=axes[0]
    if columns:
        values=gates[columns].mean();ax.bar(range(len(values)),values)
        ax.set_xticks(range(len(values)),values.index,rotation=40,ha='right')
    ax.set_ylabel('Actual state scalar');ax.set_title(evidence+' — gates')
    branch_path=root/'branch_norms.csv'
    if branch_path.exists():
        branches=pd.read_csv(branch_path)
        columns=[c for c in branches if c.endswith('_norm')]
        values=branches[columns].mean();axes[1].bar(range(len(values)),values)
        axes[1].set_xticks(range(len(values)),values.index,rotation=40,ha='right')
        axes[1].set_title('Measured branch norms');axes[1].set_ylabel('Mean token norm')
    fig.tight_layout();fig.savefig(figures/'gates.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,('scale_audit','future_audit')):
        table=pd.read_csv(root/(name+'.csv'))
        if name=='future_audit':table=table[table.mutation.isin(['pulse','NaN','Inf'])]
        columns=[c for c in ('normalized','delivered_amplitude','encoder_adapter','mdn','probabilities') if c in table]
        values=table[columns].max()
        ax.bar(range(len(values)),values);ax.set_xticks(range(len(values)),values.index,rotation=35,ha='right')
        ax.set_yscale('symlog',linthresh=1e-8);ax.set_title(name);ax.set_ylabel('Max absolute change')
    fig.suptitle(evidence);fig.tight_layout();fig.savefig(figures/'interventions.png',dpi=160);plt.close(fig)
    (figures/'CAPTIONS.md').write_text('''# 诊断图注 / Diagnostic captions

证据类型以图标题为准。SYNTHETIC 表示合成输入与未训练的小型下游；
PhaseNet/EQT 的 encoder 可使用已认证 STEAD 权重，仍不是生产 checkpoint 结论。
查询流图展示各段的跨查询 RMS；不同表示量纲不同，不直接跨段比较绝对数值。
gate/范数图来自实际加载状态与本次前向；未来扰动图仅认证 HDF 后的前缀。
尺度图的 ON 幅值允许变化；OFF 保留时长。低能量 clamp 限制另见 eps_limit JSON。

Evidence: '''+evidence+'''. Source data: ../query_stages.csv, ../gates.csv,
../scale_audit.csv and ../future_audit.csv. Arrays: ../traces/*/query_trace_arrays.npz.

1. query_flow.png: centered RMS across real query coordinates at each representation.
   Representation units differ; compare changes within a stage and K controls,
   not absolute sizes across unrelated representations. A single key makes pure
   attention constant; residual outputs can remain query dependent.
2. gates.png: values read from the actual loaded state, not config defaults;
   branch token norms use ../branch_norms.csv when present.
   Branch norms are separately recorded in traces/*/query_gates_branches.json.
3. interventions.png: measured maximum errors under positive gain and future
   pulse/NaN/Inf interventions. ON delivered amplitude may change; OFF should
   retain duration only. Last-legal-sample controls are excluded from the future
   invariance bars. HDF-prefix checks do not certify upstream preprocessing.
''')
