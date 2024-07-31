from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from tensorflow.compiler.xla.service.hlo_pb2 import HloModuleProto, HloComputationProto, HloInstructionProto

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value):
            if " " in value:
                raise ValueError("No spaces allowed in values")
            return str.__new__(cls, value)


class Color(StrEnum):
    BLUE = "#bbdefb"
    BROWN = "#bcaaa4"
    DARK_BLUE = "#1565c0"
    DARK_GREEN = "#2e7d32"
    DARK_ORANGE = "#ffb74d"
    DARK_RED = "#b71c1c"
    GRAY = "#cfd8dc"
    GREEN = "#c8e6c9"
    ORANGE = "#ffe0b2"
    PURPLE = "#e1bee7"
    RED = "#ffcdd2"
    WHITE = "#ffffff"
    YELLOW = "#fff9c4"


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
        hlo_module = HloModuleProto()
        with open(model_path, "rb") as f:
            hlo_module.ParseFromString(f.read())
        graphs = [_to_graph(comp) for comp in hlo_module.computations]
        return {"graphs": graphs}


def _to_graph_node_label(inst: HloInstructionProto) -> str:
    if inst.opcode == "parameter":
        return f"Parameter {inst.parameter_number}"
    opcode = inst.opcode
    if inst.name.startswith(opcode):
        return inst.name
    opcode += inst.fusion_kind if inst.opcode == "fusion" else ""
    return f"{opcode}{inst.name}"


def _add_incoming_edges(node: graph_builder.GraphNode, inst: HloInstructionProto):
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


def _is_effectively_scalar(inst: HloInstructionProto) -> bool:
    return (
        hasattr(inst, "shape") and hasattr(inst.shape, "dimensions") and sum(d > 1 for d in inst.shape.dimensions) == 0
    )


def _is_fused(inst: HloInstructionProto) -> bool:
    return hasattr(inst, "fused")


def _to_bg_color(inst: HloInstructionProto) -> str:
    if inst.opcode == "parameter":
        return Color.DARK_ORANGE
    if inst.opcode == "reduce-precision":
        return Color.RED
    if inst.opcode in ("convolution", "dot", "fft", "triangular-solve", "cholesky"):
        return Color.DARK_BLUE
    if inst.opcode in ("scatter", "copy", "cope-start", "copy-done"):
        return Color.GREEN
    if inst.opcode in ("dynamic-update-sclice"):
        return Color.WHITE if _is_effectively_scalar(inst) else Color.GREEN
    if inst.opcode in (
        "batch-norm-grad",
        "batch-norm-inference",
        "batch-norm-training",
        "reduce",
        "reduce-window",
        "select-and-scatter",
    ):
        return Color.PURPLE
    if inst.opcode in ("domain", "fusion", "map", "get-dimension-size", "set-dimension-size"):
        return Color.GRAY
    if inst.opcode in (
        "all-gather",
        "all-gather-start",
        "all-gather-done",
        "all-reduce",
        "reduce-scatter",
        "all-reduce-start",
        "all-reduce-done",
        "all-to-all",
        "collective-permute",
        "collective-permute-start",
        "collective-permute-done",
        "infeed",
        "outfeed",
        "partition-id",
        "recv",
        "recv-done",
        "send",
        "send-done",
        "replica-id",
    ):
        return Color.BROWN
    if inst.opcode in ("bitcast", "get-tuple-element", "trace", "after-all", "add-dependency", "tuple"):
        return Color.WHITE
    if _is_effectively_scalar(inst):
        return Color.WHITE
    if inst.opcode == "broadcast":
        return Color.WHITE if _is_effectively_scalar(inst) else Color.GREEN
    if inst.opcode in (
        "concatenate",
        "dynamic-slice",
        "gather",
        "pad",
        "reshape",
        "dynamic-range",
        "reverse",
        "tuple-select",
        "transpose",
    ):
        return Color.WHITE if _is_effectively_scalar(inst) or _is_fused(inst) else Color.GREEN
    if inst.opcode in ("call", "conditional", "custom_call", "while"):
        return Color.DARK_GREEN
    return Color.WHITE if _is_effectively_scalar(inst) else Color.YELLOW


def _to_graph_node_style(inst: HloInstructionProto) -> graph_builder.GraphNodeStyle:
    style = graph_builder.GraphNodeStyle(backgroundColor=_to_bg_color(inst))
    return style


def _to_graph_nodes(inst: HloInstructionProto) -> graph_builder.Graph:
    node = graph_builder.GraphNode(
        id=inst.id, label=_to_graph_node_label(inst), namespace="", style=_to_graph_node_style(inst)
    )
    _add_incoming_edges(node, inst)
    return node


def _to_graph(comp: HloComputationProto) -> graph_builder.Graph:
    graph = graph_builder.Graph(id=comp.name, nodes=[_to_graph_nodes(inst) for inst in comp.instructions])
    return graph
