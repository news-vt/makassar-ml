import makassar_ml.time2vec
import pytest
import torch

def test_forward():

    batch_dim = 32
    sample_size = 16
    input_dim = 6
    embed_dim = 512
    act_func = torch.sin
    t2v = makassar_ml.time2vec.Time2Vec(
        input_dim=input_dim,
        embed_dim=embed_dim, 
        act_func=act_func,
        )
    t = torch.randn(batch_dim, sample_size, input_dim)
    v = t2v(t)
    assert v.shape == torch.Size([batch_dim, sample_size, embed_dim])