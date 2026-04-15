import torch
from model.gatv2 import GATv2Layer, GATv2

def test_gatv2_layer_output_shape():
    """Single GATv2 layer produces correct output shape."""
    num_nodes = 10
    in_dim = 64
    out_dim = 128
    edge_feat_dim = 128
    num_heads = 4

    layer = GATv2Layer(in_dim, out_dim, edge_feat_dim, num_heads=num_heads)
    node_feat = torch.randn(num_nodes, in_dim)
    edges = torch.randint(0, num_nodes, (20, 2))
    edge_feat = torch.randn(20, edge_feat_dim)

    out = layer(node_feat, edges, edge_feat)
    assert out.shape == (num_nodes, out_dim), f"Expected ({num_nodes}, {out_dim}), got {out.shape}"


def test_gatv2_layer_attention_varies():
    """GATv2 attention weights should differ per edge (dynamic attention)."""
    layer = GATv2Layer(32, 64, 0, num_heads=2)
    node_feat = torch.randn(4, 32)
    node_feat[1] *= 5.0
    edges = torch.tensor([[1,0],[2,0],[3,0],[0,1],[0,2],[0,3]])

    out = layer(node_feat, edges, edge_feat=None)
    assert not torch.allclose(out[0], torch.zeros(64), atol=1e-6)


def test_gatv2_multi_layer_stack():
    """Multi-layer GATv2 stack with residual connections."""
    model = GATv2(
        node_state_dim=64,
        edge_feat_dim=128,
        num_heads=4,
        num_layer=3,
        has_residual=True
    )
    node_feat = torch.randn(15, 64)
    edges = torch.randint(0, 15, (40, 2))
    edge_feat = torch.randn(40, 128)

    out = model(node_feat, edges, edge_feat)
    assert out.shape == (15, 64)


def test_gatv2_empty_edges():
    """GATv2 handles isolated nodes (no edges) gracefully."""
    layer = GATv2Layer(32, 64, 0, num_heads=2)
    node_feat = torch.randn(5, 32)
    edges = torch.zeros(0, 2).long()

    out = layer(node_feat, edges, edge_feat=None)
    assert out.shape == (5, 64)
