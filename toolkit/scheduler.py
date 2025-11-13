import torch
from typing import Optional
from timm.scheduler.cosine_lr import CosineLRScheduler
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


class _TimmCosineWithRestartsScheduler(CosineLRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs):
        if 'total_iters' in kwargs:
            kwargs.setdefault('t_initial', kwargs.pop('total_iters'))
        if 'max_iterations' in kwargs:
            kwargs.setdefault('t_initial', kwargs.pop('max_iterations'))
        if 'T_0' in kwargs:
            kwargs.setdefault('t_initial', kwargs.pop('T_0'))
        if 'T_mult' in kwargs:
            kwargs.setdefault('cycle_mul', kwargs.pop('T_mult'))
        if 'eta_min' in kwargs:
            kwargs.setdefault('lr_min', kwargs.pop('eta_min'))

        if 't_initial' not in kwargs:
            raise ValueError("t_initial (or total_iters/max_iterations/T_0) must be provided for cosine_with_restarts scheduler")

        kwargs.setdefault('t_in_epochs', False)
        super().__init__(optimizer, **kwargs)
        self._num_updates = -1

    def step(self, epoch: Optional[int] = None, metric: Optional[float] = None):
        if epoch is None:
            self._num_updates += 1
        else:
            self._num_updates = epoch
        return super().step_update(self._num_updates, metric=metric)

    def step_update(self, num_updates: Optional[int] = None, metric: Optional[float] = None):
        if num_updates is None:
            self._num_updates += 1
        else:
            self._num_updates = num_updates
        return super().step_update(self._num_updates, metric=metric)

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif name == "cosine_with_restarts":
        return _TimmCosineWithRestartsScheduler(optimizer, **kwargs)
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
