import torch
from torch import nn
import torch.nn.functional as F

MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, img2_pred, img2_gt, gamma=0.8, max_flow=MAX_FLOW):

    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (valid >= 0.5) & (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()

    # a = 40
    # sig_epe = torch.nn.functional.sigmoid(a * (epe - 1)) # Penalise EPE > 1 pixel
    # npe_loss = (valid[:, None] * sig_epe).mean()
    # total_loss = flow_loss + 0.5 * npe_loss

    epe = epe.view(-1)[valid.view(-1)]

    img_loss = (img2_pred - img2_gt).abs().mean()
    total_loss = flow_loss + 0.5 * img_loss

    metrics = {
        'EPE': epe.mean().item(),
        '1PE': (epe > 1).float().mean().item(),
        '2PE': (epe > 2).float().mean().item(),
        '3PE': (epe > 3).float().mean().item(),
        'Loss/photometric' : img_loss.item(),
        'Loss/flow': flow_loss.item(),
        'Loss/total' : total_loss.item()
    }

    return total_loss, metrics

# def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
#     """ Loss function defined over sequence of flow predictions """
#     n_predictions = len(flow_preds)
#     flow_loss = 0.0

#     # # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
#     valid = (valid >= 0.5) & (mag < max_flow)#b,1,h,w

#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         i_loss = (flow_preds[i] - flow_gt).abs()
#         flow_loss += i_weight * (valid[:, None] * i_loss).mean()


#     epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    
#     epe = epe.view(-1)[valid.view(-1)]

#     metrics = {
#         'Flow/EPE': epe.mean().item(),
#         'Flow/1PE': (epe > 1).float().mean().item(),
#         'Flow/2PE': (epe > 2).float().mean().item(),
#         'Flow/3PE': (epe > 3).float().mean().item(),
#         'Flow/Loss': flow_loss.item(),
#     }

#     return flow_loss, metrics