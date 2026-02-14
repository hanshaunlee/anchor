"""
HGT baseline: Heterogeneous Graph Transformer (Hu et al., WWW 2020).
Type-aware attention; relative temporal encoding on edges.
Ref: https://arxiv.org/abs/2003.01332
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


class HGTBaseline(nn.Module):
    """
    HGT for node-level risk scoring on heterogeneous Independence Graph.
    Uses PyG HGTConv with multi-head type-dependent attention.
    """

    def __init__(
        self,
        in_channels: dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        metadata: tuple[list[str], list[tuple[str, str, str]]] | None = None,
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.out_channels = out_channels
        node_types = list(in_channels.keys())
        if metadata is None:
            edge_types = [
                ("person", "uses", "device"),
                ("session", "has", "utterance"),
                ("utterance", "expresses", "intent"),
                ("utterance", "mentions", "entity"),
                ("entity", "co_occurs", "entity"),
            ]
            edge_types = [e for e in edge_types if e[0] in node_types and e[2] in node_types]
            metadata = (node_types, edge_types)
        self.metadata = metadata
        node_types_list, edge_types_list = metadata

        self.lin_dict = nn.ModuleDict()
        for nt in node_types_list:
            self.lin_dict[nt] = Linear(in_channels[nt], hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    metadata=metadata,
                    heads=heads,
                )
            )
        self.lin_out = nn.ModuleDict()
        for nt in node_types_list:
            self.lin_out[nt] = Linear(hidden_channels, out_channels)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Project to hidden
        h_dict = {nt: self.lin_dict[nt](x_dict[nt]) for nt in x_dict}
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: v.relu() for k, v in h_dict.items()}
        out = {nt: self.lin_out[nt](h_dict[nt]) for nt in x_dict}
        return out

    def forward_hetero_data(self, data: HeteroData) -> dict[str, torch.Tensor]:
        # PyG 2.7: node_stores/edge_stores are sequences of storage objects; use try/except to read by key
        x_dict = {}
        for nt in self.metadata[0]:
            try:
                store = data[nt]
                if getattr(store, "x", None) is not None:
                    x = store.x
                    # Ensure feature dim matches this node type's input (e.g. graph may omit time encoding)
                    lin = self.lin_dict[nt]
                    want = getattr(lin, "in_channels", getattr(lin, "in_features", lin.weight.shape[1]))
                    have = x.size(-1)
                    if have != want:
                        if have < want:
                            x = torch.nn.functional.pad(x, (0, want - have))
                        else:
                            x = x[..., :want]
                    x_dict[nt] = x
            except (KeyError, TypeError):
                pass
        # Ensure all metadata node types are present (conv expects all keys); use empty (0, in_dim) if missing
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for nt in self.metadata[0]:
            if nt not in x_dict:
                lin = self.lin_dict[nt]
                in_dim = getattr(lin, "in_channels", getattr(lin, "in_features", lin.weight.shape[1]))
                x_dict[nt] = torch.empty(0, in_dim, device=device, dtype=dtype)
        edge_index_dict = {}
        for (src, rel, dst) in self.metadata[1]:
            key = (src, rel, dst)
            try:
                store = data[key]
                if getattr(store, "edge_index", None) is not None:
                    edge_index_dict[key] = store.edge_index
            except (KeyError, TypeError):
                pass
        return self.forward(x_dict, edge_index_dict)
