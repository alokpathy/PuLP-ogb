import argparse
import dgl
import ogb
import torch

from ogb.nodeproppred import DglNodePropPredDataset

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

d_name = args.dataset
dataset = DglNodePropPredDataset(name=d_name)

graph = dataset[0][0]
edges_src = graph.edges()[0].to(torch.int32)
edges_dst = graph.edges()[1].to(torch.int32)

print(edges_src)
print(edges_dst)

graph_data = torch.stack((edges_src, edges_dst), dim=1).view(edges_src.size(0) * 2)

print(graph_data)

# graph = torch.IntTensor(6)
# graph[0] = 0
# graph[1] = 1
# graph[2] = 1
# graph[3] = 2
# graph[4] = 2
# graph[5] = 3

my_values = {
    'graph': graph_data
}

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(my_values))
container.save("{}.pt".format(d_name))
