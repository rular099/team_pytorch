"""Numeric-store-only dynamic figures. No map download, no inferred ground truth."""
import html
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from PIL import Image
from .contracts import EPOCHS, contract as default_contract
from .io import csv_read,csv_write,fingerprint,read_json,require,write_json


def render_frame(station,grid,contract=None,title='',dpi=85):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    c=contract or default_contract()
    families=[f for f in EPOCHS if f in set(station.family)]
    require(bool(families),'No station rows for renderer')
    fig,axes=plt.subplots(3,len(families),figsize=(4*len(families),8),squeeze=False,layout='constrained')
    # Identical event bounds across all models and all cutoffs; caller passes fixed query panel.
    all_lat=station.latitude.to_numpy()
    all_lon=station.longitude.to_numpy()
    if len(grid):
        all_lat=np.r_[all_lat,grid.latitude]
        all_lon=np.r_[all_lon,grid.longitude]
    pad=.03
    bounds=[float(all_lon.min()-pad),float(all_lon.max()+pad),float(all_lat.min()-pad),float(all_lat.max()+pad)]
    for col,family in enumerate(families):
        real=station[station.family.eq(family)]
        g=grid[grid.family.eq(family)] if len(grid) else pd.DataFrame()
        supported=g[g.status.eq('supported')] if len(g) else pd.DataFrame()
        for row,(field,limits,label) in enumerate([
                ('prediction',c['render']['pga_log_color_limits'],'final log10 PGA (m/s²)'),
                ('error',c['render']['error_log_color_limits'],'direct-station prediction - truth (dex)'),
                ('width95',c['render']['width95_color_limits'],'MDN 95% interval width (dex)')]):
            ax=axes[row,col]
            data=real.assign(error=real.prediction-real.truth) if field=='error' or not len(supported) else supported
            if field not in data:
                data=data.assign(**{field:np.nan})
            color=ax.scatter(data.longitude,data.latitude,c=data[field],s=13 if data is real else 9,
                             cmap='coolwarm' if field=='error' else 'viridis',vmin=limits[0],vmax=limits[1])
            if row != 1 and len(g) and field in g and g[['latitude','longitude']].drop_duplicates().shape[0]==len(g):
                surface=g.pivot(index='latitude',columns='longitude',values=field).sort_index().sort_index(axis=1)
                color=ax.pcolormesh(surface.columns.to_numpy(),surface.index.to_numpy(),
                                    np.ma.masked_invalid(surface.to_numpy(dtype=float)),shading='nearest',
                                    cmap='viridis',vmin=limits[0],vmax=limits[1],zorder=0)
            if row==0 and len(supported)>=3:
                try:
                    ax.tricontour(supported.longitude,supported.latitude,supported.prediction,
                                  levels=c['render']['contour_log_levels'],colors='black',linewidths=.4)
                except (ValueError,RuntimeError):
                    pass  # contour geometry can be degenerate; scatter remains the truthful display
            if row==0:
                edges=plt.get_cmap('viridis')(Normalize(*limits)(real.truth.to_numpy()))
                ax.scatter(real.longitude,real.latitude,facecolors='none',edgecolors=edges,s=45,linewidths=1.6,
                           label='ring: retrospective final truth')
            if len(g):
                unsupported=g[g.status.ne('supported')]
                ax.scatter(unsupported.longitude,unsupported.latitude,c='lightgrey',marker='x',s=6)
                if 'within_100km_input_support' in supported:
                    outside=supported[~supported.within_100km_input_support.astype(bool)]
                    ax.scatter(outside.longitude,outside.latitude,facecolors='none',edgecolors='grey',marker='s',s=9,linewidths=.4)
            for coords in real.input_coords.drop_duplicates() if 'input_coords' in real else []:
                inp=np.asarray(json.loads(coords))
                if len(inp):
                    ax.scatter(inp[:,1],inp[:,0],marker='^',s=35,facecolors='none',edgecolors='red',label='received inputs')
            ax.set(xlim=bounds[:2],ylim=bounds[2:],xlabel='Longitude (degrees)',ylabel='Latitude (degrees)')
            ax.set_aspect(1/max(.1,np.cos(np.deg2rad(float(np.mean(all_lat))))))
            fig.colorbar(color,ax=ax,label=label,extend='both')
            if row==0:
                checkpoint=real.checkpoint_epoch.iloc[0] if 'checkpoint_epoch' in real else '?'
                ready=real.get('numeric_map_ready_seconds',pd.Series(dtype=float)).dropna()
                timing=f'\nmap numeric ready: firstP+{float(ready.max()):.2f}s' if len(ready) else '\nmap delivery clock: not measured/unknown'
                ax.set_title(f'{family} ep{checkpoint} K={real.input_count.iloc[0] if "input_count" in real else "?"} Q={len(real)}'+timing,fontsize=9)
                spacing=g.grid_spacing_km.dropna().iloc[0] if len(g) and 'grid_spacing_km' in g and g.grid_spacing_km.notna().any() else None
                support_note=('grid spacing '+str(spacing)+'km; height estimated\nno grid truth; rings: retrospective truth') if spacing is not None and len(supported) else 'grid predictions; spacing in source manifest\nno grid truth; rings: retrospective truth' if len(supported) else 'STATION-SUPPORTED DISPLAY\nno available grid prediction'
                ax.text(.01,.01,support_note,
                        transform=ax.transAxes,fontsize=6,bbox=dict(facecolor='white',alpha=.8))
        if 'event_latitude' in real:
            axes[0,col].scatter(real.event_longitude.iloc[0],real.event_latitude.iloc[0],marker='*',c='red',s=50)
    fig.suptitle(title+'\nAll frames predict the same FINAL PGA; retrospective truth/catalog overlays are unavailable online',fontsize=10)
    bio=io.BytesIO()
    fig.savefig(bio,format='png',dpi=dpi)
    plt.close(fig)
    bio.seek(0)
    return Image.open(bio).convert('RGB')


def animate_event(station,grid,out,ttum=None,synthetic=False):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    frame_dir=out/'frames'
    frame_dir.mkdir()
    paths=[]
    timing=[]
    c=default_contract()
    extra=set()
    if ttum is not None and len(ttum):
        for t in ttum.stable_confirmation.dropna():
            extra.update([int(t)-1,int(t),int(t)+1])
    for t in range(1,21):
        part=station[station.elapsed_time.eq(t)]
        require(len(part)>0,'Missing fixed time panel must have placeholder rows')
        g=grid[grid.elapsed_time.eq(t)] if len(grid) else grid
        start=time.perf_counter()
        origin=part.get('origin_elapsed_seconds',pd.Series(dtype=float)).dropna()
        o=f'; origin+{float(origin.iloc[0]):.2f}s' if len(origin) else '; origin clock UNKNOWN'
        title=('SYNTHETIC — NOT JAPAN | ' if synthetic else '')+str(part.event_id.iloc[0])+f' | firstP+{t}s'+o+'\nFrame indexed by data cutoff; numeric delivery per model when measured; communication UNKNOWN'
        if 'current_sample' in part:
            title+=' | inclusive data cutoff sample '+str(int(part.current_sample.iloc[0]))
        img=render_frame(part,g,c,title)
        path=frame_dir/f'{t:02d}.png'
        img.save(path)
        timing.append(dict(elapsed_time=t,render_publish_seconds=time.perf_counter()-start))
        paths.append(path)
        if t in c['render']['keyframes_first_p_seconds'] or t in extra:
            high=render_frame(part,g,c,title,dpi=125)
            high.save(out/f'keyframe_{t:02d}.png')
    animation='animation.gif'
    if shutil.which('ffmpeg'):
        result=subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-framerate','2','-start_number','1',
                               '-i',str(frame_dir/'%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                               str(out/'animation.mp4')],capture_output=True,text=True)
        if result.returncode==0:
            animation='animation.mp4'
        else:
            write_json(out/'ffmpeg_failure.json',dict(returncode=result.returncode,reason=result.stderr))
    if animation.endswith('.gif'):
        images=[Image.open(p).convert('P',palette=Image.Palette.ADAPTIVE) for p in paths]
        images[0].save(out/animation,save_all=True,append_images=images[1:],duration=500,loop=0)
    markup='''<!doctype html><meta charset="utf-8"><title>FE01 dynamic final PGA</title>
<style>body{font-family:sans-serif}img{max-width:100%}video{max-width:100%}</style>
<h1>Final PGA dynamic replay</h1><p>研究展示；真值和目录为回顾性图层。不是当前累计峰值。网格没有连续地面真值。</p>
<p>色标、等值线、范围在事件内固定。灰叉：缺网格高程；灰框：距输入站100km外的显示支撑标记，不是误差保证。</p>
<input id="slider" type="range" min="1" max="20" value="1"><span id="time">1s</span><br><img id="frame" src="frames/01.png">
<script>slider.oninput=()=>{time.textContent=slider.value+'s';frame.src='frames/'+slider.value.padStart(2,'0')+'.png'}</script>'''
    if synthetic:
        markup='<p><strong>SYNTHETIC — NOT JAPAN RESULTS</strong></p>'+markup
    markup+=f'<p><a href="{animation}">Animation</a> | <a href="frame_sources.json">Frame source contract</a></p>'
    (out/'index.html').write_text(markup)
    csv_write(pd.DataFrame(timing),out/'render_latency.csv')
    write_json(out/'frame_sources.json',dict(frames=[str(p.relative_to(out)) for p in paths],
        times_first_p_seconds=list(range(1,21)),animation=animation,playback_seconds_per_frame=.5,
        physical_update_seconds=1,contract_render=c['render'],synthetic=synthetic,
        station_scores='direct model queries at true station coordinates; no grid interpolation scoring',
        third_coordinate='km station height; metadata_estimated inside station hull only; rest explicitly unsupported'))
    return dict(directory=str(out),animation=animation,frames=20)


def shadow_map_html(outcomes,path,synthetic=False):
    cols=['family','panel','threshold_logpga','p_alarm','tau','scene','station_id','latitude','longitude','status','lead','affected']
    records=outcomes[cols].replace({np.nan:None}).to_dict('records')
    raw=json.dumps(records,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    text='''<!doctype html><meta charset="utf-8"><title>Station warning classification</title>
<h1>Station shadow / false alarms — retrospective research</h1>
<p>共同真实站点，全部阈值敏感性。面积 NA：没有独立共同等积空间支撑。不是连续区域实测盲区。</p>
<div id="controls"></div><canvas id="map" width="900" height="600"></canvas><pre id="legend"></pre>
<script>const rows=DATA, keys=['family','panel','threshold_logpga','p_alarm','tau','scene'], controls={};
for(const key of keys){let s=document.createElement('select');for(const v of [...new Set(rows.map(r=>r[key]))]){let o=document.createElement('option');o.value=v;o.textContent=key+': '+v;s.append(o)}controls[key]=s;s.onchange=draw;document.getElementById('controls').append(s)}
function draw(){let r=rows.filter(r=>keys.every(k=>String(r[k])===controls[k].value)),c=document.getElementById('map').getContext('2d');c.clearRect(0,0,900,600);if(!r.length)return;
let x=r.map(r=>r.longitude),y=r.map(r=>r.latitude),xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...y),ymax=Math.max(...y),counts={};
let colors={timely:'green',positive_but_short:'orange',late_or_zero:'red',false_alarm:'purple',true_negative:'blue'};
for(const p of r){c.beginPath();c.arc(30+840*(p.longitude-xmin)/(xmax-xmin||1),570-540*(p.latitude-ymin)/(ymax-ymin||1),5,0,7);c.fillStyle=colors[p.status]||'gray';c.fill();counts[p.status]=(counts[p.status]||0)+1}
document.getElementById('legend').textContent=JSON.stringify({status_counts:counts,colors,unknown:'gray',axes:'longitude / latitude; all real station coordinates'},null,2)}draw();</script>'''
    Path(path).write_text(('SYNTHETIC — NOT JAPAN\n' if synthetic else '')+text.replace('DATA',raw))


def build_index(entries,path,title='FE01 V4 full event visual index'):
    rows='\n'.join('<li><a href="'+html.escape(e['href'],quote=True)+'">'+html.escape(e['label'])+'</a></li>' for e in entries)
    Path(path).write_text('<!doctype html><meta charset="utf-8"><h1>'+html.escape(title)+'</h1><p>Scope as named above; metadata-selected or complete population, never success-filtered.</p><ul>'+rows+'</ul>')
