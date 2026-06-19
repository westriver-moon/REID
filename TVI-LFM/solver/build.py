import torch
from torch.optim import _functional as optim_F

from .lr_scheduler import LRSchedulerWithWarmup


class AdamWSkipEmptyGrad(torch.optim.AdamW):
    """AdamW compatible with older PyTorch builds that fail on empty grad groups."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                params_with_grad.append(param)
                if param.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads.append(param.grad)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            if not params_with_grad:
                continue

            beta1, beta2 = group["betas"]
            optim_F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad,
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )

        return loss


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')

    def is_visual_param(name):
        return name.startswith("base_model.visual") or ".visual." in name

    def is_text_param(name):
        return (
            name.startswith("base_model.transformer")
            or name.startswith("base_model.token_embedding")
            or name.startswith("base_model.positional_embedding")
            or name.startswith("base_model.ln_final")
            or name.startswith("base_model.text_projection")
        )

    def is_pmt_backbone_param(name):
        return (
            name.startswith("base_model.visual.vit.patch_embed")
            or name.startswith("base_model.visual.vit.blocks")
        )
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr_visual
        weight_decay = args.visual_weight_decay
        
        if is_visual_param(key):
            lr = args.lr_visual
            if getattr(args, "pmt_recipe", False) and is_pmt_backbone_param(key):
                lr = args.lr_visual * getattr(args, "pmt_backbone_lr_factor", 0.5)
            if "bias" in key:
                lr = args.lr_visual * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias
                if getattr(args, "pmt_recipe", False) and is_pmt_backbone_param(key):
                    lr = args.lr_visual * getattr(args, "pmt_backbone_lr_factor", 0.5) * args.visual_bias_lr_factor
            if "cross" in key:
                # use large learning rate for random initialized cross modal module
                lr =  args.lr_visual * args.lr_factor # default 5.0

        elif is_text_param(key):
            lr = args.lr_txt
            if "bias" in key:
                lr = args.lr_txt * args.text_bias_lr_factor
                weight_decay = args.text_weight_decay_bias
            if "cross" in key:
                lr =  args.lr_txt * args.lr_factor # default 5.0

        elif "classifier" in key or "mcm_head" in key or "mlm_head" in key:
                lr = args.lr_visual * args.classifier_lr_factor
        
        elif "bias" in key:
                lr = args.lr_visual * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias

        elif "cross" in key:
                # use large learning rate for random initialized cross modal module
                lr =  args.lr_visual * args.lr_factor # default 5.0

        group = {"params": [value], "lr": lr, "weight_decay": weight_decay}
        if args.optimizer in ["Adam", "AdamW"]:
            group["betas"] = (args.alpha, args.beta)
            group["eps"] = 1e-8
        params.append(group)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam": # default
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr_visual,
            weight_decay=args.visual_weight_decay,
            betas=(args.alpha, args.beta),
            eps=1e-8, # 1e-3 --> 1e-8 !!!
        )
    elif args.optimizer == "AdamW":
        optimizer = AdamWSkipEmptyGrad(
            params,
            lr=getattr(args, "lr", args.lr_visual),
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.total_train_epoch, # 120
        mode=args.lrscheduler, # useless
        target_lr=args.target_lr, # useless
        target_lr_factor=getattr(args, "target_lr_factor", None),
        power=args.power, # useless
    )
