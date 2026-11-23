"""Legacy MDN ordering/softplus semantics, with separately visible loss terms."""
import math
import torch
from torch.nn import functional as F


def nll(prediction, truth, dimensions):
    mu = prediction[...,1:1+dimensions]
    sigma = prediction[...,1+dimensions:]
    z = (truth.unsqueeze(-2)-mu)/sigma
    density = (-.5*z.square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1)
    return -torch.logsumexp(density+prediction[...,0].log_softmax(-1),-1)


def loss_terms(outputs, labels, valid, cfg):
    mag, loc, pga = outputs
    truth_mag, truth_loc, truth_pga = labels
    norm, weights = cfg['target_normalization'], cfg['loss']
    normalized = (truth_pga-norm['mean'])/norm['std']
    raw = dict(pga_nll=nll(pga,normalized,1)[valid].mean(),
               location=nll(loc,truth_loc,3).mean(),magnitude=nll(mag,truth_mag,1).mean())
    mean = (pga[...,0].softmax(-1)*pga[...,1]).sum(-1)
    # Same coordinates and smooth_l1 beta as FE02 point_regression_loss_full.
    pga_huber = F.smooth_l1_loss(mean[valid],normalized[...,0][valid],beta=weights['huber_delta'])
    loc_mean = (loc[...,0].softmax(-1)[...,None]*loc[...,1:4]).sum(-2)
    mag_mean = mag[:,0,1:2]
    loc_huber = F.smooth_l1_loss(loc_mean,truth_loc,beta=weights['huber_delta'])
    mag_huber = F.smooth_l1_loss(mag_mean,truth_mag,beta=weights['huber_delta'])
    # Legacy mixture_density_loss_full divides task weights by their sum.
    denominator = weights['pga_nll']+weights['location']+weights['magnitude']
    raw['mean_huber'] = (pga_huber*weights['pga_nll']+loc_huber*weights['location']+
                         mag_huber*weights['magnitude'])/denominator
    weighted = {name:value*weights[name]/denominator for name,value in raw.items() if name != 'mean_huber'}
    weighted['mean_huber'] = raw['mean_huber']*weights['mean_huber']
    total = sum(weighted.values())
    return total, {**{k+'_raw':v for k,v in raw.items()}, **{k+'_weighted':v for k,v in weighted.items()}}
