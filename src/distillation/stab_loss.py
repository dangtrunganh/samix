import torch


def compute_ird(features, temp):
    features1_sim = torch.div(torch.matmul(features, features.T), temp)
    logits_mask = torch.scatter(
        torch.ones_like(features1_sim), 1,
        torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True), 0)

    logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
    features1_sim = features1_sim - logits_max1.detach()
    row_size = features1_sim.size(0)

    logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
        features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
    return logits1


def compute_sprd(features, prototypes, temp):
    features1_sim = torch.div(torch.matmul(features, prototypes.T), temp)
    logits_max1, _ = torch.max(features1_sim, dim=1, keepdim=True)
    features1_sim = features1_sim - logits_max1.detach()
    return torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=1, keepdim=True)


def compute_distillation_loss(opt, features1_prev_task, frozen_model, images, set_prototypes=None,
                              alpha_balance_distillation=None):
    if opt.distillation == 'ird':
        logits1 = compute_ird(features1_prev_task, opt.current_temp)
        with torch.no_grad():
            features2_prev_task = frozen_model(images)
            logits2 = compute_ird(features2_prev_task, opt.past_temp)
        return (-logits2 * torch.log(logits1)).sum(1).mean()

    elif opt.distillation == 'sprd':
        assert set_prototypes is not None
        logits1 = compute_sprd(features1_prev_task, set_prototypes, opt.current_temp)
        with torch.no_grad():
            features2_prev_task = frozen_model(images)
            logits2 = compute_sprd(features2_prev_task, set_prototypes, opt.past_temp)
        return (-logits2 * torch.log(logits1)).sum(1).mean()

    elif opt.distillation == 'hsd':
        assert set_prototypes is not None
        assert alpha_balance_distillation is not None
        current_temp_sprd = opt.current_temp
        past_temp_sprd = opt.past_temp
        if hasattr(opt, "current_temp_sprd") and hasattr(opt, "past_temp_sprd"):
            if opt.current_temp_sprd is not None and opt.past_temp_sprd is not None:
                current_temp_sprd = opt.current_temp_sprd
                past_temp_sprd = opt.past_temp_sprd
        logits1 = compute_ird(features1_prev_task, opt.current_temp)
        logits1_proto = compute_sprd(features1_prev_task, set_prototypes, current_temp_sprd)
        with torch.no_grad():
            features2_prev_task = frozen_model(images)
            logits2 = compute_ird(features2_prev_task, opt.past_temp)
            logits2_proto = compute_sprd(features2_prev_task, set_prototypes, past_temp_sprd)
        return (1 - alpha_balance_distillation) * (-logits2 * torch.log(logits1)).sum(
            1).mean() + alpha_balance_distillation * (-logits2_proto * torch.log(logits1_proto)).sum(1).mean()
