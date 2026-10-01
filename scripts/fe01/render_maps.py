#!/usr/bin/env python
"""Render actual frame outputs: grid predictions, observed points, reference and residuals."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.spatial import reference_design
from fe01.config import sha256,write_json

def render(source,reference_path,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    source=Path(source);destination=Path(destination)
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(destination)
    destination.mkdir(parents=True,exist_ok=True)
    stations=pd.read_csv(source/'replay_predictions.csv.gz',dtype={'event_id':str,'station_id':str})
    nodes=pd.read_csv(source/'grid_predictions.csv.gz',dtype={'event_id':str})
    support=pd.read_csv(source/'replay_support.csv',dtype={'event_id':str})
    reference=json.loads(Path(reference_path).read_text())
    coef=np.asarray(reference['no_site_coefficients']);data=[]
    if not stations.split.eq('val').all(): raise ValueError('Maps are locked validation replay only')
    for event_id,frames in support.groupby('event_id'):
        images=[];last=None
        for entry in frames.itertuples():
            elapsed=entry.elapsed_time
            figure,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
            axes=axes.ravel()
            if entry.status!='supported':
                for axis in axes: axis.axis('off')
                axes[0].text(.1,.5,f'{entry.status}: {entry.reason}',transform=axes[0].transAxes)
            else:
                obs=stations.loc[stations.event_id.eq(event_id)&stations.elapsed_time.eq(elapsed)].copy()
                mesh=nodes.loc[nodes.event_id.eq(event_id)&nodes.elapsed_time.eq(elapsed)].copy()
                if not len(obs) or not len(mesh): raise ValueError('Supported frame missing actual observations/grid')
                for name in ['event_latitude','event_longitude','depth','magnitude']: mesh[name]=obs[name].iloc[0]
                mesh['reference']=reference_design(mesh)@coef
                mesh['residual']=mesh.prediction-mesh.reference
                mesh['width95']=mesh.q975-mesh.q025
                obs['reference']=reference_design(obs)@coef
                obs['observed_residual']=obs.truth-obs.reference
                obs['error']=obs.prediction-obs.truth
                obs.to_csv(destination/f'{event_id}_t{elapsed:03d}_stations.csv.gz',index=False)
                data.append(mesh)
                panels=[(mesh,'prediction','Grid mixture mean',-3,1),(obs,'truth','Observed final PGA (points)',-3,1),
                    (obs,'prediction','Station predictions',-3,1),(mesh,'reference','Train-only oracle attenuation',-3,1),
                    (mesh,'residual','Predicted minus reference',-1,1),(obs,'observed_residual','Observed minus reference',-1,1),
                    (obs,'error','Station prediction error',-1,1),(mesh,'width95','Mixture 95% interval width',0,2)]
                inputs=np.asarray(json.loads(obs.input_coords.iloc[0]))
                for axis,(table,field,title,low,high) in zip(axes,panels):
                    valid=table.coverage.astype(str).str.lower().eq('true') if 'coverage' in table else np.ones(len(table),bool)
                    artist=axis.scatter(table.longitude[valid],table.latitude[valid],c=table[field][valid],s=14,
                        vmin=low,vmax=high,cmap='coolwarm' if low==-1 else 'viridis')
                    if 'coverage' in table:
                        axis.scatter(table.longitude[~valid],table.latitude[~valid],marker='x',s=8,color='lightgray')
                    axis.scatter(inputs[:,1],inputs[:,0],s=38,facecolors='none',edgecolors='black',linewidths=.8)
                    axis.set(title=title,xlabel='Longitude',ylabel='Latitude')
                    axis.set_aspect(1/np.cos(np.deg2rad(obs.latitude.mean())))
                    figure.colorbar(artist,ax=axis,shrink=.7,label='log10(m/s²)' if field!='width95' else 'log10 width')
            figure.suptitle(f'{event_id} | t={elapsed}s | Every frame predicts the same final PGA; input cutoff changes')
            path=destination/f'{event_id}_t{elapsed:03d}.png';figure.savefig(path,dpi=100)
            if elapsed in [1,3,5,10,20,40,90]: figure.savefig(path.with_suffix('.pdf'))
            plt.close(figure)
            with Image.open(path) as image: images.append(image.convert('P',palette=Image.Palette.ADAPTIVE))
        if images:
            images[0].save(destination/f'{event_id}.gif',save_all=True,append_images=images[1:],duration=250,loop=0)
    if data: pd.concat(data).to_csv(destination/'grid_figure_data.csv.gz',index=False)
    write_json(destination/'map_provenance.json',dict(replay_provenance_sha256=sha256(source/'replay_provenance.json'),
        reference_sha256=sha256(reference_path),split='val',pga_colors=[-3,1],residual_colors=[-1,1],width_colors=[0,2],
        coverage='heuristic <=100km to input; gray crosses outside',truth='observed stations only; never interpolated',
        reference='retrospective final catalog source parameters; train-only coefficients',
        animation='actual inference frames, unsupported frames blank, no interpolation'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();render(a.replay,a.reference,a.output)
