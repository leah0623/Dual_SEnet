import torch
import torch.nn as nn

class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_loss(self, P, T, E, M, mode):
        P_exp = torch.exp(P) # (B,)
        P_exp_B = torch.stack([P_exp for _ in range(P.shape[0])], dim=0) # (B, B)
        if mode == 'risk':
            E = E.float() * (M.sum(dim=1) > 0).float()
        elif mode == 'surv':
            E = (M.sum(dim=1) > 0).float()
        else:
            raise NotImplementedError
        P_exp_sum = (P_exp_B * M.float()).sum(dim=1)
        P_tmp = P_exp / (P_exp_sum+1e-6)
        loss = -torch.sum(torch.log(P_tmp.clip(1e-6, P_tmp.max().item())) * E) / torch.sum(E)
        return loss

    def forward(self, P_risk, T, E):
        # P: (B,)
        # T: (B,)
        # E: (B,) \in {0, 1}
        M_risk = T.unsqueeze(dim=1) < T.unsqueeze(dim=0) # (B, B)
        loss_risk = self._compute_loss(P_risk, T, E, M_risk, mode='risk')
        return loss_risk

class mydeepsurvival(nn.Module):
    def __init__(self):
        super(mydeepsurvival, self).__init__()

    def forward(self, Prisk, event_times, event):
        sort_idx = torch.argsort(event_times, descending=True)
        risk_sorted = Prisk[sort_idx]
        event_sorted = event[sort_idx]

        log_sum = torch.logcumsumexp(risk_sorted, dim=0)
        likelihood = risk_sorted - log_sum
        censored_likelihood = likelihood * event_sorted
        numevents = event_sorted.sum()
        if numevents == 0:
            return torch.tensor(0.0, device=Prisk.device, requires_grad=Prisk.requires_grad)
        else:
            return -censored_likelihood.sum() / numevents

class weighted_surv_loss(nn.Module):
    def __init__(self):
        super(weighted_surv_loss,self).__init__()

    def forward(self, Prisk, event_times, event, weight_ratio=0.15, l2_reg=0.01):
        # l2_term = l2_reg * torch.sum(torch.square(Prisk))

        sort_idx = torch.argsort(event_times, descending=True)
        risk_sorted = Prisk[sort_idx]
        event_sorted = event[sort_idx]

        weight = torch.where(event==1, 1, weight_ratio)
        log_sum = torch.logcumsumexp(risk_sorted, dim=0)
        likelihood = risk_sorted - log_sum
        censored_likelihood = likelihood * event_sorted * weight
        weightnumevent = event_sorted * weight
        numevents = weightnumevent.sum()
        if numevents == 0:
            return torch.tensor(0.0, device=Prisk.device, requires_grad=Prisk.requires_grad)
        else:
            return -censored_likelihood.sum() / numevents
