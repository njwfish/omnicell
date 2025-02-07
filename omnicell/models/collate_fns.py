import torch
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,OTPlanSampler
)

import numpy as np


def fm_collate(
    batch,
    return_noise=False,
    return_control=False,
    FM=ExactOptimalTransportConditionalFlowMatcher(sigma=0.1),
):
    noises = []

    control, pert = zip(*batch)
    control = torch.tensor(np.array(control))
    pert = torch.tensor(np.array(pert))

    if return_noise:
        t, xt, ut, eps = FM.sample_location_and_conditional_flow(
            control,
            pert,
            return_noise=return_noise,
        )
        noises.append(eps)
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(
            control,
            pert,
            return_noise=return_noise,
        )
    ret = [t, xt, ut]
    if return_noise:
        noises = torch.cat(noises)
        ret.append(noises)
    if return_control:
        ret.append(control)
    return ret


def cfm_collate(
    batch,
    return_noise=False,
    return_control=False,
    FM=ExactOptimalTransportConditionalFlowMatcher(sigma=0.1),
):
    noises = []

    control, target, perturb = zip(*batch)
    control = torch.tensor(np.array(control))
    target = torch.tensor(np.array(target))
    perturb = torch.tensor(np.array(perturb))

    if return_noise:
        t, xt, ut, eps = FM.sample_location_and_conditional_flow(
            control,
            target,
            return_noise=return_noise,
        )
        noises.append(eps)
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(
            control,
            target,
            return_noise=return_noise,
        )
    ret = [t, xt, ut, perturb]
    if return_noise:
        noises = torch.cat(noises)
        ret.append(noises)
    if return_control:
        ret.append(control)
    return ret


def ot_collate(
    batch,
    return_noise=False,
    ot_sampler = OTPlanSampler(method="exact")
):

    batch = list(zip(*batch))
    batch = [torch.tensor(np.array(x)) for x in batch]
    control, target = batch[:2]
    
    x0, x1 = ot_sampler.sample_plan(control, target)

    return x0, x1, *batch[2:]
