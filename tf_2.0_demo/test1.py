import torch
import torchtext


# %%

t = torch.tensor([[1., -1.], [1., -1.]])
t
t.reshape([])


# %%

torchtext.vocab.GloVe(name='6B', dim=50)




# %%
