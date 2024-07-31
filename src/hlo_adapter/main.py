from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from tensorflow.compiler.xla.service import hlo_pb2


class HloAdapter(Adapter):
    metadata = AdapterMetadata(
        id="hlo_adapter",
        name="XLA HLO adapter",
        description="An adapter to explore XLA HLO",
        source_repo="https://github.com/clumsy/hlo-adapter",
        fileExts=["hlo"],
    )

    # Required.
    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        hlo_module = hlo_pb2.HloModuleProto()
        with open(model_path, "rb") as f:
            hlo_module.ParseFromString(f.read())
        graphs = [_to_graph(comp) for comp in hlo_module.computations]
        return {"graphs": graphs}


def _to_graph_node_label(inst: hlo_pb2.HloInstructionProto) -> str:
    if inst.opcode == "parameter":
        return f"Parameter {inst.parameter_number}"
    opcode = inst.opcode
    if inst.name.startswith(opcode):
        return inst.name
    opcode += inst.fusion_kind if inst.opcode == "fusion" else ""
    return f"{opcode}{inst.name}"


def _add_incoming_edges(node: graph_builder.GraphNode, inst: hlo_pb2.HloInstructionProto):
    if inst.opcode == "parameter" and hasattr(inst, "fused"):
        pass
    else:
        for i, op_id in enumerate(inst.operand_ids):
            node.incomingEdges.append(graph_builder.IncomingEdge(sourceNodeId=op_id, targetNodeInputId=str(i)))
        if hasattr(inst, "control_predecessors"):
            for pred in inst.control_predecessors:
                node.incomingEdges.append(
                    graph_builder.IncomingEdge(sourceNodeId=pred.id, targetNodeInputId="0")
                )  # mark as control edge


def _to_bg_color(inst: hlo_pb2.HloInstructionProto) -> str:
    if inst.opcode == "parameter":
        return "#ffb74d"
    if inst.opcode == "broadcast":
        return "#fff9c4"
    if inst.opcode == "reduce":
        return "#e1bee7"
    if inst.opcode == "all-gather" or inst.opcode == "reduce-scatter":
        return "#bcaaa4"
    return ""


def _to_graph_node_style(inst: hlo_pb2.HloInstructionProto) -> graph_builder.GraphNodeStyle:
    style = graph_builder.GraphNodeStyle(backgroundColor=_to_bg_color(inst))
    return style


def _to_graph_nodes(inst: hlo_pb2.HloInstructionProto) -> graph_builder.Graph:
    node = graph_builder.GraphNode(
        id=inst.id, label=_to_graph_node_label(inst), namespace="", style=_to_graph_node_style(inst)
    )
    _add_incoming_edges(node, inst)
    return node


def _to_graph(comp: hlo_pb2.HloComputationProto) -> graph_builder.Graph:
    graph = graph_builder.Graph(id=comp.name, nodes=[_to_graph_nodes(inst) for inst in comp.instructions])
    return graph
