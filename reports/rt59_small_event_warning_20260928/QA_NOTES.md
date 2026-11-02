# Figure and analysis QA

- Backend exclusivity: all data analysis, drawing, preview and export used
  Python; no R graphics device was used.
- Split: validation only; held-out test was not opened.
- Model: immutable RT59 same-forward reference exported in the RT61 validation
  NPZ; RT61 candidate predictions were not used as the evaluated model.
- Small-event definition: catalog magnitude at or below 3.0. All 23 eligible
  validation events are exactly magnitude 3.0; no smaller event is present.
- Replicate definition: confidence intervals resample event IDs as clusters.
- Metric coordinate: log10(m/s²).
- Low support: joint cells with fewer than 10 unique events are hatched and
  marked with a dagger; their line-panel points are hollow and unconnected.
- Editable export: SVG keeps text as text nodes; PDF uses TrueType font
  embedding; PNG is a 300-dpi preview.
- Color: sequential blue/purple/gray maps; no rainbow or red-green-only coding.
- Image integrity: no microscopy, photographic adjustment or raster source
  manipulation is involved.
