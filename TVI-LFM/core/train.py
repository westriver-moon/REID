import torch
from tools import MultiItemAverageMeter
from torch.cuda import amp


def train(base, loaders, scaler, config, optimizer):
    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    if config.pretrain_choice == "LASTVIT_ORI":
        mode = "1/3"
    elif "RN" in config.pretrain_choice:
        mode = "1/3"
    elif "ViT" in config.pretrain_choice:
        mode = None
    else:
        raise ValueError(f"Pretrain model {config.pretrain_choice} choice not supported")

    for i, batch_dict in enumerate(loader):
        batch_dict = {key: value.to(base.device) for key, value in batch_dict.items()}
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(enabled=True):
            ret = base(batch_dict, mode)
            losses = [value for key, value in ret.items() if 'loss' in key]
            if not losses:
                raise RuntimeError("No loss terms were produced by model forward")
            total_loss = sum(losses)

        if not torch.isfinite(total_loss):
            if getattr(config, 'skip_non_finite_batches', True):
                print(f"[warn] skip non-finite loss at iter {i}: {float(total_loss.detach().cpu())}")
                continue
            raise FloatingPointError(f"Non-finite loss at iter {i}: {float(total_loss.detach().cpu())}")

        scaler.scale(total_loss).backward()

        grad_clip_norm = float(getattr(config, 'grad_clip_norm', 0.0) or 0.0)
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        acc_sign = False
        acc_value = 0
        for key, value in ret.items():
            if 'loss' in key:
                meter.update({key: value})
            if 'acc' in key:
                acc_sign = True
                acc_value = value
        meter.update({'total_loss': total_loss})
        if acc_sign:
            meter.update({'acc': acc_value})

        if config.train_max_iter > 0 and (i + 1) >= config.train_max_iter:
            break

    return meter.get_val(), meter.get_str()
