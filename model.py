from torch_geometric.nn.conv import GATv2Conv#, HGTConv
from torch_geometric.nn import Linear
from torch_geometric.nn import to_hetero
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, SparseTensor, Size, Any
from torch_geometric.utils import softmax, is_sparse


def construct_bipartite_edge_index(
    edge_index_dict: Dict[EdgeType, Adj],
    src_offset_dict: Dict[EdgeType, int],
    dst_offset_dict: Dict[NodeType, int],
    edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Adj, Optional[Tensor]]:

    is_sparse_tensor = False
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    for edge_type, src_offset in src_offset_dict.items():
        edge_index = edge_index_dict[edge_type]
        dst_offset = dst_offset_dict[edge_type[-1]]

        # TODO Add support for SparseTensor w/o converting.
        #print(edge_index, SparseTensor)
        is_sparse_tensor = isinstance(edge_index, SparseTensor)
        if is_sparse(edge_index):
            edge_index, _ = to_edge_index(edge_index)
            edge_index = edge_index.flip([0])
        else:
            edge_index = edge_index.clone()

        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_indices.append(edge_index)

        if edge_attr_dict != None:
            if isinstance(edge_attr_dict, ParameterDict):
                value = edge_attr_dict['__'.join(edge_type)]
            else:
                value = edge_attr_dict[edge_type]
            if value.size(0) != edge_index.size(1):
                value = value.expand(edge_index.size(1), -1)
            edge_attrs.append(value)

    edge_index = torch.cat(edge_indices, dim=1)

    edge_attr: Optional[Tensor] = None
    if edge_attr_dict != None:
        edge_attr = torch.cat(edge_attrs, dim=0)

    if is_sparse_tensor:
        edge_index = SparseTensor(
            row=edge_index[1],
            col=edge_index[0],
            value=edge_attr,
            sparse_sizes=(num_nodes, num_nodes),
        )

    return edge_index, edge_attr



class HGTConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = set([key[-1] for key in self.edge_types])

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)


    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset


    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def propagate(
      self,
      edge_index: Adj,
      size: Size = None,
      **kwargs: Any,
      ) -> Tensor:

      decomposed_layers = 1 if self.explain else self.decomposed_layers

      print(self._propagate_forward_pre_hooks.values())
      for hook in self._propagate_forward_pre_hooks.values():
          res = hook(self, (edge_index, size, kwargs))
          if res != None:
              edge_index, size, kwargs = res

      mutable_size = self._check_input(edge_index, size)

      # Run "fused" message and aggregation (if applicable).
      fuse = False
      if self.fuse and not self.explain:
          if is_sparse(edge_index):
              fuse = True
          elif (not torch.jit.is_scripting()
                and isinstance(edge_index, EdgeIndex)):
              if (self.SUPPORTS_FUSED_EDGE_INDEX
                      and edge_index.is_sorted_by_col):
                  fuse = True

      if fuse:
          coll_dict = self._collect(self._fused_user_args, edge_index,
                                    mutable_size, kwargs)

          msg_aggr_kwargs = self.inspector.collect_param_data(
              'message_and_aggregate', coll_dict)
          for hook in self._message_and_aggregate_forward_pre_hooks.values():
              res = hook(self, (edge_index, msg_aggr_kwargs))
              if res != None:
                  edge_index, msg_aggr_kwargs = res
          out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
          for hook in self._message_and_aggregate_forward_hooks.values():
              res = hook(self, (edge_index, msg_aggr_kwargs), out)
              if res != None:
                  out = res

          update_kwargs = self.inspector.collect_param_data(
              'update', coll_dict)
          out = self.update(out, **update_kwargs)

      else:  # Otherwise, run both functions in separation.
          if decomposed_layers > 1:
              user_args = self._user_args
              decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
              decomp_kwargs = {
                  a: kwargs[a].chunk(decomposed_layers, -1)
                  for a in decomp_args
              }
              decomp_out = []

          for i in range(decomposed_layers):
              if decomposed_layers > 1:
                  for arg in decomp_args:
                      kwargs[arg] = decomp_kwargs[arg][i]

              coll_dict = self._collect(self._user_args, edge_index,
                                        mutable_size, kwargs)

              msg_kwargs = self.inspector.collect_param_data(
                  'message', coll_dict)
              for hook in self._message_forward_pre_hooks.values():
                  res = hook(self, (msg_kwargs, ))
                  if res != None:
                      msg_kwargs = res[0] if isinstance(res, tuple) else res
              out = self.message(**msg_kwargs)
              for hook in self._message_forward_hooks.values():
                  res = hook(self, (msg_kwargs, ), out)
                  if res != None:
                      out = res

              if self.explain:
                  explain_msg_kwargs = self.inspector.collect_param_data(
                      'explain_message', coll_dict)
                  out = self.explain_message(out, **explain_msg_kwargs)

              aggr_kwargs = self.inspector.collect_param_data(
                  'aggregate', coll_dict)
              for hook in self._aggregate_forward_pre_hooks.values():
                  res = hook(self, (aggr_kwargs, ))
                  if res != None:
                      aggr_kwargs = res[0] if isinstance(res, tuple) else res

              out = self.aggregate(out, **aggr_kwargs)

              for hook in self._aggregate_forward_hooks.values():
                  res = hook(self, (aggr_kwargs, ), out)
                  if res != None:
                      out = res

              update_kwargs = self.inspector.collect_param_data(
                  'update', coll_dict)
              out = self.update(out, **update_kwargs)

              if decomposed_layers > 1:
                  decomp_out.append(out)

          if decomposed_layers > 1:
              out = torch.cat(decomp_out, dim=-1)

      for hook in self._propagate_forward_hooks.values():
          res = hook(self, (edge_index, mutable_size, kwargs), out)
          if res != None:
              out = res

      return out


    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        #print(kqv_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        # print(edge_index_dict)
        edge_index_used = torch.cat([edge_index_dict[key] for key in edge_index_dict], dim=1) # cat all tensors in edge_index_dict to shape [2, num_edges]

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        print(edge_index.shape, edge_index_used.shape, edge_attr)
        out = self.propagate(edge_index_used, k=k, q=q, v=v, edge_attr=edge_attr)

        #print(edge_index.shape, k.permute(2, 1, 0).shape, q.permute(0, 2, 1).shape, v.permute(2, 1, 0).shape, k.size(2))

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v != None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict


    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')



class HGT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_classes=3, heads=1, num_layers=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        # node_type: linear layer
        self.lin_dict = torch.nn.ModuleDict()

        # List of HGTConv layers
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
          conv = HGTConv(
              in_channels=hidden_channels,
              out_channels=hidden_channels,
              metadata=metadata,
              heads=heads,
          )
          self.convs.append(conv)

        self.classifier = nn.Linear(hidden_channels, num_classes)


    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        # print([v.shape for k, v in x_dict.items()])
        print(len(edge_index_dict))

        for node_type, x in x_dict.items():
            out_channels = x.size(0)
            layer = Linear(out_channels, self.hidden_channels)
            x_dict[node_type] = layer(x.T).relu()

        #print([v.shape for k, v in x_dict.items()])
        #print(x_dict)

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.classifier(x_dict)
