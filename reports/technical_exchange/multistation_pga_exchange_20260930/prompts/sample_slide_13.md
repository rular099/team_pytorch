# Slide 13 — RT61 truth-versus-prediction evidence sample

Use case: productivity-visual / scientific evidence slide.

Create exactly one complete landscape PowerPoint slide, 16:9, at the highest practical resolution. All Chinese text must be clear, correctly spelled and crisp. This is an internal technical report about seismic peak ground acceleration (PGA), not a marketing infographic.

Input image 1 is a STRICT EVIDENCE ASSET: `assets/figures/rt61_scatter_four_populations.png`. Insert the supplied four-panel chart as one intact image. Preserve every plotted point, contour, fitted line, axis limit, tick, label, title, sample count and numerical value. Scale uniformly. Do not redraw, reinterpret, remove outliers, invent points, retouch data or crop away any chart content. This scientific chart was computed directly from authenticated formal prediction exports.

Input image 2 is STYLE ONLY: the old deck's `slide_11.png`. Borrow its light grey-white background, dark charcoal Chinese sans-serif typography, thin muted blue title rule, muted blue/teal palette, subtle pale-blue card borders, and evidence-first scientific layout. Do not copy its content, metric table, old scatter plot, icon set or dense layout.

Composition: generous 16:9 scientific slide. A short dark-charcoal title across the top, a thin blue horizontal rule underneath. The supplied four-panel evidence figure occupies the left approximately 65–68% of the slide's body, large and readable with all labels intact. The right approximately 27–30% is a clean vertical evidence column with three flat pale cards. Keep a narrow gutter, sufficient white space and no overlapping text. The bottom has a slim takeaway band and one small source/protocol line. No title clipping, no page number, no logo, no decorative pictograms, no map, no waveforms, no gradients, no drop shadows.

Exact title:
"真实 PGA 与预测 PGA——输入站更准，远程目标仍有压缩"

Small protocol tag under the title or at top-right:
"RT61 · epoch 8 · seed 42 · validation"

Right card 1, blue accent:
"输入站预测最准确"
"MAE  0.0945"
"R²  0.8405"
"已观测波形提供直接约束"

Right card 2, teal accent:
"远程预测更困难"
"Normal non-input MAE  0.1992"
"Random geometry MAE  0.2373"
"应用评价需单列非输入目标"

Right card 3, restrained coral accent:
"动态范围仍被压缩"
"Random slope  0.4113"
"Non-input slope  0.5154"
"高 PGA 目标仍有低估"

Small legend underneath the three cards:
"灰色虚线：1:1 参考线"
"珊瑚实线：线性拟合"
"等高线：50% / 80% / 95% 密度范围"

Bottom takeaway band, readable and concise:
"结论：点预测已改善；随机几何与非输入目标的空间反差仍需加强。"

Bottom source/protocol line:
"Japan 2000–2024 固定验证集｜预测为 log10 PGA 坐标上的 MDN 均值｜全量有效点，透明度 0.08｜统一轴域"

Scientific invariants: all four panels are RT61 candidate predictions from epoch-8 validation. Random n=75,654; normal all n=89,770; normal input n=63,651; normal non-input n=26,119. Never describe this as test performance. The formal metrics inside the supplied chart must stay exact: random MAE 0.2373, R² 0.3994, slope 0.4113; normal all MAE 0.1250, R² 0.7388, slope 0.7446; normal input MAE 0.0945, R² 0.8405, slope 0.8257; normal non-input MAE 0.1992, R² 0.4223, slope 0.5154. Do not add a metric not listed here. The chart's x/y coordinates are log10(m/s²); preserve this label.
