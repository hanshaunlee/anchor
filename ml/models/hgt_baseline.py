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
                ("session", "has_event", "event"),
                ("event", "mentions", "entity"),
                ("event", "next_event", "event"),
                ("utterance", "expresses", "intent"),
                ("utterance", "mentions", "entity"),
                ("entity", "co_occurs", "entity"),
            ]
            edge_types = [e for e in edge_types if e[0] in node_types and e[2] in node_types]
            metadata = (node_types, edge_types)
        self.metadata = metadata
        node_types_list, edge_types_list = metadata
        # Store expected input dim per node type (for padding in forward_hetero_data)
        self._in_channels = {nt: int(in_channels[nt]) for nt in node_types_list}

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

    def _ensure_hidden(self, h_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return a dict with keys in metadata order and every value with last dim exactly self.hidden."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        out = {}
        for nt in self.metadata[0]:
            t = h_dict.get(nt, torch.empty(0, self.hidden, device=device, dtype=dtype))
            if t.size(-1) != self.hidden:
                if t.size(-1) > self.hidden:
                    t = t[..., : self.hidden].contiguous()
                else:
                    t = torch.nn.functional.pad(t, (0, self.hidden - t.size(-1)))
            out[nt] = t
        return out

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Project to hidden
        h_dict = {nt: self.lin_dict[nt](x_dict[nt]) for nt in x_dict}
        h_dict = self._ensure_hidden(h_dict)
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: v.relu() for k, v in h_dict.items()}
            h_dict = self._ensure_hidden(h_dict)
        out = {nt: self.lin_out[nt](h_dict[nt]) for nt in x_dict}
        return out

    def forward_hetero_data(self, data: HeteroData) -> dict[str, torch.Tensor]:
        # PyG 2.7: node_stores/edge_stores are sequences of storage objects; use try/except to read by key
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x_dict = {}
        for nt in self.metadata[0]:
            want = self._in_channels.get(nt)
            if want is None:
                continue
            try:
                store = data[nt]
                if getattr(store, "x", None) is not None:
                    x = store.x
                    if x.device != device:
                        x = x.to(device=device, dtype=dtype)
                    have = x.size(-1)
                    if have != want:
                        if have < want:
                            x = torch.nn.functional.pad(x, (0, want - have))
                        else:
                            x = x[..., :want].contiguous()
                    x_dict[nt] = x
            except (KeyError, TypeError):
                pass
        # Ensure all metadata node types are present (conv expects all keys); use empty (0, in_dim) if missing
        for nt in self.metadata[0]:
            if nt not in x_dict:
                in_dim = self._in_channels.get(nt)
                if in_dim is not None:
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

    def forward_with_hidden(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run forward and return (logits_dict, hidden_dict). Hidden is the pooled representation before lin_out (for use as incident embeddings)."""
        h_dict = {nt: self.lin_dict[nt](x_dict[nt]) for nt in x_dict}
        h_dict = self._ensure_hidden(h_dict)
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: v.relu() for k, v in h_dict.items()}
            h_dict = self._ensure_hidden(h_dict)
        out = {nt: self.lin_out[nt](h_dict[nt]) for nt in x_dict}
        return out, h_dict

    def forward_hetero_data_with_hidden(self, data: HeteroData) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Same as forward_hetero_data but returns (logits_dict, hidden_dict) for embedding extraction."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x_dict = {}
        for nt in self.metadata[0]:
            want = self._in_channels.get(nt)
            if want is None:
                continue
            try:
                store = data[nt]
                if getattr(store, "x", None) is not None:
                    x = store.x
                    if x.device != device:
                        x = x.to(device=device, dtype=dtype)
                    have = x.size(-1)
                    if have != want:
                        if have < want:
                            x = torch.nn.functional.pad(x, (0, want - have))
                        else:
                            x = x[..., :want].contiguous()
                    x_dict[nt] = x
            except (KeyError, TypeError):
                pass
        for nt in self.metadata[0]:
            if nt not in x_dict:
                in_dim = self._in_channels.get(nt)
                if in_dim is not None:
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
        return self.forward_with_hidden(x_dict, edge_index_dict)
