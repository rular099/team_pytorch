#!/usr/bin/env python3
"""One OFF optimizer update per accessible family on synthetic train/val data."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--weights',required=True)
    args=p.parse_args();os.environ['FE01_A01_TEST_WEIGHTS']=args.weights
    import torch
    from test_fe01_a01_helpers import config,sample
    from fe01.engine import loss
    from fe01.model import common_state,state_fingerprint
    from fe01_a01 import OFF
    from fe01_a01.model import build_model
    from fe01_a01.provenance import new_output,write_json,require,source_identity
    output=new_output(args.output);rows=[]
    for family in ('team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen'):
        with tempfile.TemporaryDirectory(prefix='fe01-a01-off-smoke-') as tmp:
            cfg=config(Path(tmp),family,OFF);event,pack,inputs=sample(Path(tmp),cfg)
            model=build_model(cfg);model.train()
            before=state_fingerprint(model.state_dict());common_before=state_fingerprint(common_state(model))
            encoder_before=state_fingerprint(model.waveform_model.encoder.state_dict())
            labels=[x.unsqueeze(0) for x in pack['labels']]
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],weight_decay=0.)
            value=loss(model(*inputs),labels,model,cfg,inputs[4]);require(torch.isfinite(value),'Nonfinite OFF smoke loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip']);opt.step()
            after=state_fingerprint(model.state_dict())
            require(before!=after,'OFF smoke updated no parameters')
            if family!='team_original_scratch':
                require(encoder_before==state_fingerprint(model.waveform_model.encoder.state_dict()) and
                    not model.waveform_model.encoder.training,'Frozen encoder changed')
            rows.append(dict(family=family,mode=OFF,status='PASS',loss=float(value.detach()),
                initial_state_sha256=before,updated_state_sha256=after,common_initial_sha256=common_before,
                optimizer_steps=1,scope='SYNTHETIC one microbatch, not formal batch128/DDP training',
                frozen_encoder_unchanged=family!='team_original_scratch',initialization='factory initial state, not final ON'))
    write_json(output/'OFF_single_update_smoke.json',dict(rows=rows,source=source_identity(),formal='NOT_SUBMITTED'))


if __name__=='__main__':main()
