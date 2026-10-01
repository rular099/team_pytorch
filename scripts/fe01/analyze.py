#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.analysis import compare_runs,fit_train_reference,summarize_run,training_histograms
from fe01.config import load

def main():
    p=argparse.ArgumentParser(description='FE01 held-out metrics, train-only reference and paired comparisons')
    p.add_argument('action',choices=['reference','summarize','compare','training-histograms'])
    p.add_argument('--config');p.add_argument('--evaluations',nargs='+')
    p.add_argument('--reference');p.add_argument('--run-dir');p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='reference':
        fit_train_reference(load(a.config),a.output)
    elif a.action=='summarize':
        summarize_run(a.evaluations[0],a.output,a.reference)
    elif a.action=='compare':
        compare_runs(a.evaluations,a.output)
    else:
        training_histograms(a.run_dir,a.output)
if __name__=='__main__':
    main()
