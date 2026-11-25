"""List compiled Vitis AI subgraphs and exposed output tensors."""

import json
import argparse
from pathlib import Path

import xir

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/out/compiled/wrn_l12_b512.xmodel")
path = Path(parser.parse_args().path)
graph = xir.Graph.deserialize(str(path))
root = graph.get_root_subgraph()
subgraphs = root.toposort_child_subgraph()
rows = []
for subgraph in subgraphs:
    device = subgraph.get_attr("device") if subgraph.has_attr("device") else "unknown"
    outputs = [
        {"name": tensor.name, "dims": list(tensor.dims)}
        for tensor in subgraph.get_output_tensors()
    ]
    rows.append({"name": subgraph.get_name(), "device": device, "outputs": outputs})
print(json.dumps({"graph": graph.get_name(), "subgraphs": rows}, indent=2))
