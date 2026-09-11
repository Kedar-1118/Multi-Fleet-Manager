"""City graph model for the fleet routing simulation.

Creates a NetworkX-based city representation with weighted edges
for distance and travel time, supporting shortest-path queries
and configurable graph topologies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np


@dataclass
class CityGraphConfig:
    """Configuration for city graph generation.

    Attributes:
        num_nodes: Number of intersection/service nodes.
        grid_size: City area dimension in km (grid_size x grid_size).
        edge_density: Probability of edge existence between nearby nodes.
        min_edge_weight: Minimum road distance in km.
        max_edge_weight: Maximum road distance in km.
        base_speed_kmh: Base vehicle speed in km/h.
        seed: Random seed for reproducibility.
    """
    num_nodes: int = 50
    grid_size: float = 10.0
    edge_density: float = 0.15
    min_edge_weight: float = 1.0
    max_edge_weight: float = 5.0
    base_speed_kmh: float = 30.0
    num_charging_stations: int = 3
    seed: int = 42


class CityGraph:
    """Graph-based city model for fleet routing simulation.

    Nodes represent intersections or service locations.
    Edges represent roads with distance and base travel time weights.
    The graph is guaranteed to be connected.

    Attributes:
        config: Configuration parameters for graph generation.
        graph: The underlying NetworkX graph.
        node_positions: Mapping from node ID to (x, y) coordinates.
    """

    def __init__(self, config: Optional[CityGraphConfig] = None) -> None:
        """Initialize the city graph.

        Args:
            config: Graph configuration. Uses defaults if None.
        """
        self.config = config or CityGraphConfig()
        self.graph: nx.Graph = nx.Graph()
        self.node_positions: dict[int, tuple[float, float]] = {}
        self.charging_stations: list[int] = []
        self._rng = np.random.RandomState(self.config.seed)
        self._shortest_path_cache: dict[tuple[int, int], list[int]] = {}
        self._distance_cache: dict[tuple[int, int], float] = {}
        self._build_graph()
        self._designate_charging_stations()

    def _build_graph(self) -> None:
        """Construct the city graph with nodes, edges, and weights."""
        n = self.config.num_nodes
        size = self.config.grid_size

        # Place nodes randomly in 2D space
        for i in range(n):
            x = self._rng.uniform(0, size)
            y = self._rng.uniform(0, size)
            self.node_positions[i] = (x, y)
            self.graph.add_node(i, pos=(x, y))

        # Create edges based on distance and density
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(i, j)
                # Higher probability for closer nodes
                threshold = self.config.edge_density * (size / max(dist, 0.1))
                if self._rng.random() < min(threshold, 0.8):
                    travel_time = dist / self.config.base_speed_kmh * 60  # minutes
                    self.graph.add_edge(
                        i, j,
                        distance=round(dist, 3),
                        base_travel_time=round(travel_time, 3),
                    )

        # Ensure connectivity by adding edges to form a spanning tree
        self._ensure_connectivity()

    def _ensure_connectivity(self) -> None:
        """Add minimum edges to make the graph connected."""
        if nx.is_connected(self.graph):
            return

        components = list(nx.connected_components(self.graph))
        for idx in range(len(components) - 1):
            # Find closest pair between components
            min_dist = float("inf")
            best_pair = (0, 0)
            comp_a = list(components[idx])
            comp_b = list(components[idx + 1])
            for a in comp_a:
                for b in comp_b:
                    d = self._euclidean_distance(a, b)
                    if d < min_dist:
                        min_dist = d
                        best_pair = (a, b)

            travel_time = min_dist / self.config.base_speed_kmh * 60
            self.graph.add_edge(
                best_pair[0], best_pair[1],
                distance=round(min_dist, 3),
                base_travel_time=round(travel_time, 3),
            )

    def _euclidean_distance(self, node_a: int, node_b: int) -> float:
        """Calculate Euclidean distance between two nodes.

        Args:
            node_a: First node ID.
            node_b: Second node ID.

        Returns:
            Distance in km.
        """
        xa, ya = self.node_positions[node_a]
        xb, yb = self.node_positions[node_b]
        return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

    def get_shortest_path(self, source: int, target: int) -> list[int]:
        """Find the shortest path between two nodes using cached Dijkstra.

        Args:
            source: Starting node ID.
            target: Destination node ID.

        Returns:
            Ordered list of node IDs forming the shortest path.

        Raises:
            nx.NetworkXNoPath: If no path exists.
            nx.NodeNotFound: If source or target not in graph.
        """
        key = (source, target)
        if key not in self._shortest_path_cache:
            path = nx.shortest_path(
                self.graph, source, target, weight="distance"
            )
            self._shortest_path_cache[key] = path
        return self._shortest_path_cache[key]

    def get_shortest_distance(self, source: int, target: int) -> float:
        """Get the shortest distance between two nodes.

        Args:
            source: Starting node ID.
            target: Destination node ID.

        Returns:
            Shortest distance in km.
        """
        key = (source, target)
        if key not in self._distance_cache:
            try:
                dist = nx.shortest_path_length(
                    self.graph, source, target, weight="distance"
                )
            except nx.NetworkXNoPath:
                dist = float("inf")
            self._distance_cache[key] = dist
        return self._distance_cache[key]

    def get_travel_time(
        self,
        source: int,
        target: int,
        traffic_multiplier: float = 1.0,
    ) -> float:
        """Estimate travel time between two nodes with traffic.

        Args:
            source: Starting node ID.
            target: Destination node ID.
            traffic_multiplier: Traffic condition multiplier (1.0 = normal).

        Returns:
            Estimated travel time in minutes.
        """
        distance = self.get_shortest_distance(source, target)
        if distance == float("inf"):
            return float("inf")
        base_time = distance / self.config.base_speed_kmh * 60  # minutes
        return base_time * traffic_multiplier

    def get_neighbors(self, node: int) -> list[int]:
        """Get all neighbor nodes of a given node.

        Args:
            node: Node ID.

        Returns:
            List of neighboring node IDs.
        """
        return list(self.graph.neighbors(node))

    def get_edge_distance(self, node_a: int, node_b: int) -> float:
        """Get direct edge distance between two adjacent nodes.

        Args:
            node_a: First node ID.
            node_b: Second node ID.

        Returns:
            Edge distance in km, or inf if no direct edge.
        """
        if self.graph.has_edge(node_a, node_b):
            return self.graph[node_a][node_b]["distance"]
        return float("inf")

    def get_node_position(self, node: int) -> tuple[float, float]:
        """Get the (x, y) position of a node.

        Args:
            node: Node ID.

        Returns:
            Tuple of (x, y) coordinates.

        Raises:
            KeyError: If node doesn't exist.
        """
        return self.node_positions[node]

    def get_normalized_position(self, node: int) -> tuple[float, float]:
        """Get node position normalized to [0, 1] range.

        Args:
            node: Node ID.

        Returns:
            Tuple of normalized (x, y) coordinates.
        """
        x, y = self.node_positions[node]
        return (x / self.config.grid_size, y / self.config.grid_size)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self.graph.number_of_edges()

    def get_random_node(self, rng: Optional[np.random.RandomState] = None) -> int:
        """Get a random node ID from the graph.

        Args:
            rng: Random number generator. Uses internal RNG if None.

        Returns:
            A random node ID.
        """
        gen = rng if rng is not None else self._rng
        return int(gen.choice(list(self.graph.nodes)))

    def get_random_node_pair(
        self, rng: Optional[np.random.RandomState] = None
    ) -> tuple[int, int]:
        """Get two distinct random node IDs.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of two distinct node IDs.
        """
        gen = rng if rng is not None else self._rng
        nodes = list(self.graph.nodes)
        pair = gen.choice(nodes, size=2, replace=False)
        return (int(pair[0]), int(pair[1]))

    def clear_caches(self) -> None:
        """Clear shortest path and distance caches."""
        self._shortest_path_cache.clear()
        self._distance_cache.clear()

    def _designate_charging_stations(self) -> None:
        """Designate spatially-distributed nodes as EV charging stations.

        Uses a greedy farthest-point strategy to ensure charging stations
        are spread across the city rather than clustered together.
        """
        n_stations = min(
            self.config.num_charging_stations, self.config.num_nodes
        )
        if n_stations <= 0:
            return

        nodes = list(self.graph.nodes)
        # Start with the node closest to the city center
        center = (self.config.grid_size / 2, self.config.grid_size / 2)
        first = min(
            nodes,
            key=lambda n: math.sqrt(
                (self.node_positions[n][0] - center[0]) ** 2
                + (self.node_positions[n][1] - center[1]) ** 2
            ),
        )
        self.charging_stations = [first]
        self.graph.nodes[first]["is_charging_station"] = True

        # Greedily add the farthest node from all existing stations
        for _ in range(n_stations - 1):
            best_node = -1
            best_min_dist = -1.0
            for n in nodes:
                if n in self.charging_stations:
                    continue
                min_dist = min(
                    self._euclidean_distance(n, s)
                    for s in self.charging_stations
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_node = n
            if best_node >= 0:
                self.charging_stations.append(best_node)
                self.graph.nodes[best_node]["is_charging_station"] = True

    def is_charging_station(self, node: int) -> bool:
        """Check if a node is a designated charging station.

        Args:
            node: Node ID.

        Returns:
            True if the node is a charging station.
        """
        return node in self.charging_stations

    def get_nearest_charging_station(
        self, node: int
    ) -> tuple[int, float]:
        """Find the nearest charging station to a given node.

        Args:
            node: Source node ID.

        Returns:
            Tuple of (station_node_id, distance_km).

        Raises:
            ValueError: If no charging stations are configured.
        """
        if not self.charging_stations:
            raise ValueError("No charging stations configured in the city graph.")

        best_station = self.charging_stations[0]
        best_dist = self.get_shortest_distance(node, best_station)

        for station in self.charging_stations[1:]:
            dist = self.get_shortest_distance(node, station)
            if dist < best_dist:
                best_dist = dist
                best_station = station

        return best_station, best_dist
