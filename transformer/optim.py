from torch.optim.lr_scheduler import LambdaLR

def update_learning_rate(step_num: int, d_model: int, factor: float, warmup_steps: int = 4000) -> float:
    """
    Update learning rate based on the schedule in "Attention is All You Need".
    """
    if step_num == 0:
        step_num = 1
    return factor * (
        d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )

class OptimScheduler():
    """Wrapper for optimizer and learning rate scheduler."""
    def __init__(self, optimizer, d_model: int, factor: float = 1.0, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.lr_scheduler = LambdaLR(optimizer,
            lr_lambda=lambda step_num: update_learning_rate(step_num, self.d_model, self.factor, self.warmup_steps)
        )
    
    def step(self) -> None:
        self.optimizer.step()
        self.lr_scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
    
    def get_learning_rate(self):
        """Get the current learning rate of the optimizer."""
        return self.optimizer.param_groups[0]['lr']
