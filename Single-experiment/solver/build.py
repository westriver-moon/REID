import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')

    has_global_lastvit_pretrain = bool(getattr(args, "lastvit_pretrained", None))
    has_rgb_lastvit_pretrain = has_global_lastvit_pretrain or bool(getattr(args, "lastvit_pretrained_rgb", None))
    has_ir_lastvit_pretrain = has_global_lastvit_pretrain or bool(getattr(args, "lastvit_pretrained_ir", None))

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = args.lr_visual
        weight_decay = args.visual_weight_decay

        if "transformer" in key:
            lr = args.lr_txt
            if "bias" in key:
                lr = args.lr_txt * args.text_bias_lr_factor
                weight_decay = args.text_weight_decay_bias
            if "cross" in key:
                lr = args.lr_txt * args.lr_factor

        elif "visual" in key:
            lr = args.lr_visual
            # LASTViT random/newly introduced parameters should use boosted lr.
            is_lastvit_adapter = (
                "visual.img_projection" in key
                or "visual.ir_patch_embed" in key
                or ("visual.rgb_backbone" in key and getattr(args, "pretrain_choice", "") == "LASTVIT_ORI" and not has_rgb_lastvit_pretrain)
                or ("visual.ir_backbone" in key and getattr(args, "pretrain_choice", "") == "LASTVIT_ORI" and not has_ir_lastvit_pretrain)
            )
            if "bias" in key:
                lr = args.lr_visual * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias
            if "cross" in key or is_lastvit_adapter:
                lr = args.lr_visual * args.lr_factor
                if "bias" in key:
                    lr = args.lr_visual * args.visual_bias_lr_factor * args.lr_factor

        elif "classifier" in key or "mcm_head" in key or "mlm_head" in key:
            lr = args.lr_visual * args.classifier_lr_factor

        elif "bias" in key:
            lr = args.lr_visual * args.visual_bias_lr_factor
            weight_decay = args.visual_weight_decay_bias

        elif "cross" in key:
            lr = args.lr_visual * args.lr_factor

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = str(args.optimizer).lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr_visual, momentum=args.momentum
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr_visual,
            weight_decay=args.visual_weight_decay,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr_visual,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        raise NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.total_train_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
