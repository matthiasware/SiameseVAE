import torch
import torch_optimizer as optim


def get_optimizer(optimizer: str, model, optimizer_args):
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), **optimizer_args)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_args)
    elif optimizer == "yogi":
        return optim.Yogi(model.parameters(), **optimizer_args)
    elif optimizer == "shampoo":
        return optim.Shampoo(model.parameters(), **optimizer_args)
    elif optimizer == "swats":
        return optim.SWATS(model.parameters(), **optimizer_args)
    elif optimizer == "sgdw":
        return optim.SGDW(model.parameters(), **optimizer_args)
    elif optimizer == "sgdp":
        return optim.SGDP(model.parameters(), **optimizer_args)
    elif optimizer == "rangerva":
        return optim.RangerVA(model.parameters(), **optimizer_args)
    elif optimizer == "rangerqh":
        return optim.RangerQH(model.parameters(), **optimizer_args)
    elif optimizer == "ranger":
        return optim.Ranger(model.parameters(), **optimizer_args)
    elif optimizer == "radam":
        return optim.RAdam(model.parameters(), **optimizer_args)
    elif optimizer == "qhm":
        return optim.QHM(model.parameters(), **optimizer_args)
    elif optimizer == "qhadam":
        return optim.QHAdam(model.parameters(), **optimizer_args)
    elif optimizer == "pid":
        return optim.PID(model.parameters(), **optimizer_args)
    elif optimizer == "novograd":
        return optim.NovoGrad(model.parameters(), **optimizer_args)
    elif optimizer == "lamb":
        return optim.Lamb(model.parameters(), **optimizer_args)
    elif optimizer == "diffgrad":
        return optim.DiffGrad(model.parameters(), **optimizer_args)
    elif optimizer == "apollo":
        return optim.Apollo(model.parameters(), **optimizer_args)
    elif optimizer == "aggmo":
        return optim.AggMo(model.parameters(), **optimizer_args)
    elif optimizer == "adamp":
        return optim.AdamP(model.parameters(), **optimizer_args)
    elif optimizer == "adafactor":
        return optim.Adafactor(model.parameters(), **optimizer_args)
    elif optimizer == "adamod":
        return optim.AdaMod(model.parameters(), **optimizer_args)
    elif optimizer == "adabound":
        return optim.AdaBound(model.parameters(), **optimizer_args)
    elif optimizer == "adabelief":
        return optim.AdaBelief(model.parameters(), **optimizer_args)
    elif optimizer == "accsgd":
        return optim.AccSGD(model.parameters(), **optimizer_args)
    elif optimizer == "a2graduni":
        return optim.A2GradUni(model.parameters(), **optimizer_args)
    elif optimizer == "a2gradinc":
        return optim.A2GradInc(model.parameters(), **optimizer_args)
    elif optimizer == "a2gradexp":
        return optim.A2GradExp(model.parameters(), **optimizer_args)
    else:
        raise Exception(f"Optimizer '{optimizer}' does not exist!")


def get_scheduler(scheduler: str, optimizer, scheduler_args):
    if scheduler == "cosine_decay":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          **scheduler_args)
    else:
        raise Exception(f"Scheduler '{scheduler}' does not exist!")
