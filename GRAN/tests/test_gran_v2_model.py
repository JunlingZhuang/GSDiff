import torch
from easydict import EasyDict as edict
from model.gran_v2 import GRANv2


def make_config(num_attr_classes=5, use_gatv2=False):
    return edict({
        'device': 'cpu',
        'model': edict({
            'max_num_nodes': 20,
            'hidden_dim': 64,
            'embedding_dim': 64,
            'is_sym': True,
            'block_size': 1,
            'sample_stride': 1,
            'num_GNN_prop': 1,
            'num_GNN_layers': 2,
            'edge_weight': 1.0,
            'dimension_reduce': True,
            'has_attention': True,
            'num_canonical_order': 1,
            'num_mix_component': 5,
            'num_attr_classes': num_attr_classes,
            'use_gatv2': use_gatv2,
            'gatv2_num_heads': 4,
        })
    })


def test_gran_v2_has_attr_head():
    """GRANv2 should have output_attr head mapping hidden_dim -> num_attr_classes."""
    config = make_config(num_attr_classes=7)
    model = GRANv2(config)
    assert hasattr(model, 'output_attr')
    dummy = torch.randn(10, 64)
    logits = model.output_attr(dummy)
    assert logits.shape == (10, 7)


def test_gran_v2_sampling_returns_adj_and_attrs():
    """Sampling should return (A_list, attr_list) tuple."""
    import numpy as np
    config = make_config(num_attr_classes=5)
    model = GRANv2(config)
    model.eval()

    input_dict = {
        'is_sampling': True,
        'batch_size': 2,
        'num_nodes_pmf': np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]
                                  + [0.0] * 10, dtype=np.float32),
    }
    result = model(input_dict)
    assert isinstance(result, tuple) and len(result) == 2
    A_list, attr_list = result
    assert len(A_list) == 2
    assert len(attr_list) == 2
    for A, attrs in zip(A_list, attr_list):
        n = A.shape[0]
        assert attrs.shape == (n,), f"Expected ({n},), got {attrs.shape}"
        assert attrs.max() < 5
        assert attrs.min() >= 0


def test_gran_v2_with_gatv2_backbone():
    """GRANv2 with use_gatv2=True uses GATv2 as decoder."""
    config = make_config(use_gatv2=True)
    model = GRANv2(config)
    assert model.decoder.__class__.__name__ == 'GATv2'


def test_gran_v2_with_gru_backbone():
    """GRANv2 with use_gatv2=False uses original GNN as decoder."""
    config = make_config(use_gatv2=False)
    model = GRANv2(config)
    assert model.decoder.__class__.__name__ == 'GNN'
