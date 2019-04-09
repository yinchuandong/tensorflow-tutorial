import torch


# %%
x = torch.zeros(2, 1, 2, 1, 2)
x.squeeze().size()

x.squeeze(1).size()
x.squeeze(3).size()
# %%


# %%
