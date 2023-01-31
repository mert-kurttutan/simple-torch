import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD
import matplotlib.pyplot as plt
from pathlib import Path
import os


STEPS = 100
dir_path = Path(__file__).parent
optimizer = SGD([torch.tensor(1)], lr=1)
# Use a scheduler of your choice below.
# Great for debugging your own schedulers!
scheduler = CosineAnnealingLR(optimizer, STEPS)


scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=100, epochs=10
)

lrs = []
for _ in range(10):
    for _ in range(100):
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()




if __name__ == "__main__":
    result_path = os.path.join(dir_path, "foo.png")
    plt.plot(lrs)
    plt.savefig(result_path)
