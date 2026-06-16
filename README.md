# Delhivery Network Intelligence

A graph-based ETA optimization engine designed to move beyond traditional OSRM shortest-path heuristics. This project models the logistics network as a dynamic graph to capture real-world congestion, dwell times, and network-wide delay propagation.

### Core Approach

Traditional routing engines like OSRM assume "clean" traffic and shortest-path travel, which systematically underestimates actual delivery times in a complex logistics network. This project moves away from isolated, straight-line estimation toward a **holistic, network-aware architecture**.

* **Graph-Based Intelligence:** We model the entire logistics operation as a directed weighted graph. Facilities (hubs) function as nodes, and routes between them function as edges. Instead of using static distance, edge weights are dynamic, capturing the median actual-vs-OSRM delay ratio per corridor, stratified by route type and time of day.
* **Network Auditing:** Before predictive modeling, we conduct a mathematical audit of the network topology. Using metrics such as betweenness centrality, in-degree/out-degree, and clustering coefficients, we identify critical "chokepoint" hubs and chronically delayed corridors where actual delivery performance consistently degrades.
* **Deep Graph Learning (GraphSAGE):** We transition from manual feature engineering to advanced deep learning. Using the **GraphSAGE (Graph Sample and Aggregated)** architecture, the model learns by aggregating real-time information from a facility's neighboring nodes.
This allows the model to natively understand **delay propagation**: if a major sorting hub becomes congested, the model recognizes how this "bleeding" effect impacts the ETA of downstream trucks, even before they reach the facility.
* **Intelligent Decision Framework:** We utilize the structural embeddings learned from the GraphSAGE model to drive operational decisions. This ML-backed framework evaluates the time-cost trade-off for individual shipments, accounting for distance, time of day, and the facility's specific topological importance to recommend the optimal route type (Full Truckload vs. Carting).

### Core Objectives

* **Audit the Network:** Mathematically identify and prioritize bottleneck hubs that disproportionately contribute to SLA breaches.
* **Predictive ETA:** Deploy GNNs to produce highly accurate ETAs that account for network-wide operational reality.
* **Optimization:** Implement a data-driven framework to minimize revenue-at-risk by selecting the optimal transport mode.

### Tech Stack

* **Languages:** Python
* **Graph Frameworks:** NetworkX, PyTorch Geometric
* **ML Frameworks:** PyTorch, Pandas, NumPy
* **Version Control:** Git/GitHub

*Built to improve delivery accuracy through network-aware intelligence.*