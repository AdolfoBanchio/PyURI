import os

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

edges = [
    { "src": "PVD", "dst": "DVA", "type": "IN"},
    { "src": "PVD", "dst": "PVC", "type": "IN"},
    { "src": "PVD", "dst": "AVA", "type": "IN"},
    { "src": "PLM", "dst": "DVA", "type": "IN"},
    { "src": "PLM", "dst": "AVD", "type": "IN"},
    { "src": "PLM", "dst": "AVA", "type": "IN"},
    { "src": "PLM", "dst": "PVC", "type": "GJ"},
    { "src": "AVM", "dst": "PVC", "type": "IN"},
    { "src": "AVM", "dst": "AVD", "type": "GJ"},
    { "src": "ALM", "dst": "PVC", "type": "IN"},
    { "src": "ALM", "dst": "AVD", "type": "IN"},
    { "src": "DVA", "dst": "PVC", "type": "IN"},
    { "src": "AVD", "dst": "AVA", "type": "EX"},
    { "src": "AVD", "dst": "AVB", "type": "EX"},
    { "src": "AVD", "dst": "PVC", "type": "EX"},
    { "src": "PVC", "dst": "AVB", "type": "EX"},
    { "src": "PVC", "dst": "AVD", "type": "EX"},
    { "src": "PVC", "dst": "DVA", "type": "EX"},
    { "src": "PVC", "dst": "AVA", "type": "EX"},
    { "src": "AVA", "dst": "AVB", "type": "IN"},
    { "src": "AVA", "dst": "PVC", "type": "IN"},
    { "src": "AVA", "dst": "REV", "type": "EX"},
    { "src": "AVA", "dst": "AVD", "type": "IN"},
    { "src": "AVB", "dst": "FWD", "type": "EX"},
    { "src": "AVB", "dst": "AVA", "type": "IN"},
    { "src": "AVB", "dst": "AVD", "type": "IN"}
]

# Layering
layers = {
    0: ["PVD", "PLM", "AVM", "ALM"],          # sensory / input
    1: ["DVA", "AVD", "PVC"],                 # interneurons (stage 2)
    2: ["AVA", "AVB"],                        # interneurons (stage 3)
    3: ["REV", "FWD"]                         # motor / output
}

# Node types (for coloring)
node_type = {}
for n in layers[0]:
    node_type[n] = "sensory"
for n in layers[1] + layers[2]:
    node_type[n] = "inter"
for n in layers[3]:
    node_type[n] = "motor"

node_colors = {
    "sensory": "#ff6b6b",   # soft red
    "inter":   "#4cd37b",   # green
    "motor":   "#6fa8ff"    # blue
}
edge_colors = {
    "EX": "#2ecc71",  # green
    "IN": "#ff4d4d",  # red
    "GJ": "#3b82f6"   # blue
}

# Build graph
G = nx.DiGraph()
for layer_nodes in layers.values():
    for n in layer_nodes:
        G.add_node(n, ntype=node_type[n])

for e in edges:
    G.add_edge(e["src"], e["dst"], etype=e["type"])

# Positions: x by layer, y stacked
pos = {}
x_gap = 3.5
y_gap = 2.2

for li, nodes in layers.items():
    x = li * x_gap
    # center nodes vertically per layer
    y0 = (len(nodes) - 1) * y_gap / 2.0
    for i, n in enumerate(nodes):
        pos[n] = (x, y0 - i * y_gap)

plt.figure(figsize=(14, 8))
# Leave room underneath for legends placed outside the axis.
plt.subplots_adjust(bottom=0.22, right=0.95)

# Draw nodes by type to control colors
for t in ["sensory", "inter", "motor"]:
    nodelist = [n for n, d in G.nodes(data=True) if d["ntype"] == t]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_color=node_colors[t],
        node_size=1600,
        edgecolors="black",
        linewidths=1
    )

nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")

# Draw edges by type with arrows
# EX and IN: directed
for etype in ["EX", "IN"]:
    edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == etype]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edgelist,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        min_source_margin=18,
        min_target_margin=18,
        width=2,
        edge_color=edge_colors[etype],
        connectionstyle="arc3,rad=0.12"
    )

# GJ
gj_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "GJ"]
nx.draw_networkx_edges(
        G, pos,
        edgelist=gj_edges,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        min_source_margin=18,
        min_target_margin=18,
        width=2,
        edge_color=edge_colors["GJ"],
        connectionstyle="arc3,rad=0.12"
    )

plt.title("Tap Withdrawal Circuit (TWC)", fontsize=14)
plt.axis("off")

# Legends
node_legend = [
    Patch(facecolor=node_colors["sensory"], edgecolor="black", label="Sensory neuron"),
    Patch(facecolor=node_colors["inter"], edgecolor="black", label="Inter neuron"),
    Patch(facecolor=node_colors["motor"], edgecolor="black", label="Motor neuron"),
]
edge_legend = [
    Line2D([0], [0], color=edge_colors["EX"], lw=3, label="Excitatory (EX)"),
    Line2D([0], [0], color=edge_colors["IN"], lw=3, label="Inhibitory (IN)"),
    Line2D([0], [0], color=edge_colors["GJ"], lw=3, label="Gap junction (GJ)"),
]

leg1 = plt.legend(
    handles=node_legend,
    loc="upper center",
    bbox_to_anchor=(0.28, -0.15),
    ncol=3,
    frameon=True,
    title="Node types",
    prop={"size": 12},
    title_fontsize=12,
)
plt.gca().add_artist(leg1)
plt.legend(
    handles=edge_legend,
    loc="upper center",
    bbox_to_anchor=(0.72, -0.15),
    ncol=3,
    frameon=True,
    title="Edge types",
    prop={"size": 12},
    title_fontsize=12,
)
plt.tight_layout()

os.makedirs("out/graphs", exist_ok=True)
plt.savefig("out/graphs/twc_graph.png", dpi=300)
