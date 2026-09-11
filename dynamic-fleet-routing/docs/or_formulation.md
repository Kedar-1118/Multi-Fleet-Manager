# Formal Operations Research Formulation

## Electric Dynamic Vehicle Routing Problem with Time Windows (E-DVRPTW)

This document provides the formal mathematical programming formulation for the constrained fleet dispatch optimization problem solved by this project. The formulation is presented as a Mixed-Integer Linear Program (MILP) extended with battery State-of-Charge (SoC) constraints for electric vehicle fleets.

---

## 1. Problem Definition

### 1.1 Problem Class

The **Electric Dynamic Vehicle Routing Problem with Time Windows (E-DVRPTW)** extends the classical Capacitated Vehicle Routing Problem (CVRP) with:

- **Dynamic request arrivals** following a non-homogeneous Poisson process
- **Time windows** (SLA deadlines) on each pickup/delivery pair
- **Time-varying travel times** due to stochastic traffic conditions
- **Battery State-of-Charge constraints** with intermediate charging station visits
- **Vehicle payload capacity limits**

### 1.2 Complexity

The E-DVRPTW is **NP-hard** by reduction from the Capacitated Vehicle Routing Problem (CVRP), which is itself NP-hard by reduction from the Traveling Salesman Problem (TSP). Adding time windows, stochastic travel times, and battery constraints strictly increases computational complexity.

> **Implication**: No polynomial-time exact algorithm exists (unless P = NP). Practical solutions require either bounded exact methods (branch-and-bound, constraint programming) or approximation approaches (heuristics, metaheuristics, or learning-augmented search).

---

## 2. Sets and Indices

| Symbol | Definition |
| :--- | :--- |
| $\mathcal{V} = \{1, \ldots, K\}$ | Set of vehicles (fleet) |
| $\mathcal{N} = \{0, 1, \ldots, n, n+1\}$ | Set of nodes: $0$ and $n+1$ are depot copies |
| $\mathcal{R} = \{1, \ldots, m\}$ | Set of customer requests (pickup-delivery pairs) |
| $\mathcal{S} \subset \mathcal{N}$ | Set of charging station nodes |
| $\mathcal{E} = \{(i, j) : i, j \in \mathcal{N}, i \neq j\}$ | Set of arcs (road segments) |
| $\mathcal{T}$ | Continuous time horizon $[0, T_{\max}]$ |

---

## 3. Parameters

### 3.1 Network Parameters

| Symbol | Definition | Units |
| :--- | :--- | :--- |
| $d_{ij}$ | Shortest-path distance from node $i$ to node $j$ | km |
| $\tau_{ij}(t)$ | Time-varying travel time from $i$ to $j$ at departure time $t$ | minutes |
| $M(t, i, j)$ | Traffic multiplier on edge $(i, j)$ at time $t$ | dimensionless |

$$\tau_{ij}(t) = \frac{d_{ij}}{v_{\text{base}}} \cdot M(t, i, j) \cdot 60$$

### 3.2 Vehicle Parameters

| Symbol | Definition | Units |
| :--- | :--- | :--- |
| $Q_k$ | Payload capacity of vehicle $k$ | units |
| $B_k^{\max}$ | Battery capacity of vehicle $k$ | kWh |
| $e_k$ | Energy consumption rate of vehicle $k$ | kWh/km |
| $B_k^{\min}$ | Minimum safe battery level (low-SoC threshold) | kWh |
| $\rho$ | Charging power at fast-charging stations | kW |

### 3.3 Request Parameters

| Symbol | Definition | Units |
| :--- | :--- | :--- |
| $p_r$ | Pickup location of request $r$ | node ID |
| $q_r$ | Dropoff location of request $r$ | node ID |
| $a_r$ | Arrival (release) time of request $r$ | minutes |
| $b_r$ | Deadline (latest delivery time) of request $r$ | minutes |
| $w_r$ | Package size of request $r$ | units |
| $\pi_r$ | Priority weight of request $r$ | dimensionless |

---

## 4. Decision Variables

| Variable | Domain | Definition |
| :--- | :--- | :--- |
| $x_{ijk} \in \{0, 1\}$ | Binary | 1 if vehicle $k$ traverses arc $(i, j)$ |
| $t_{ik} \geq 0$ | Continuous | Arrival time of vehicle $k$ at node $i$ |
| $l_{ik} \geq 0$ | Continuous | Load of vehicle $k$ upon departing node $i$ |
| $b_{ik} \geq 0$ | Continuous | Battery SoC of vehicle $k$ at node $i$ |
| $y_{rk} \in \{0, 1\}$ | Binary | 1 if request $r$ is served by vehicle $k$ |
| $z_r \in \{0, 1\}$ | Binary | 1 if request $r$ is left unserved |
| $c_{ik} \geq 0$ | Continuous | Energy charged at station $i$ by vehicle $k$ |

---

## 5. Objective Function

Minimize a weighted combination of operational costs, emissions, service penalties, and deadhead distance:

$$\min \quad \sum_{k \in \mathcal{V}} \sum_{(i,j) \in \mathcal{E}} \alpha \cdot d_{ij} \cdot x_{ijk} \quad + \quad \sum_{k \in \mathcal{V}} \sum_{(i,j) \in \mathcal{E}} \beta \cdot e_k \cdot d_{ij} \cdot x_{ijk} \quad + \quad \sum_{r \in \mathcal{R}} \gamma \cdot z_r \quad + \quad \sum_{r \in \mathcal{R}} \delta \cdot \max(0, \; t_{q_r, k} - b_r) \cdot y_{rk}$$

Where:
- **Term 1** ($\alpha$): Total fleet travel distance (deadhead + service distance)
- **Term 2** ($\beta$): Total energy/carbon cost (proportional to distance × energy rate)
- **Term 3** ($\gamma$): Penalty for each unserved request
- **Term 4** ($\delta$): Tardiness penalty for late deliveries exceeding SLA deadline

---

## 6. Constraints

### 6.1 Flow Conservation

Each vehicle starts at the depot, visits a sequence of nodes, and returns:

$$\sum_{j \in \mathcal{N}} x_{0jk} = 1 \quad \forall k \in \mathcal{V}$$

$$\sum_{i \in \mathcal{N}} x_{i,n+1,k} = 1 \quad \forall k \in \mathcal{V}$$

$$\sum_{i \in \mathcal{N}} x_{ijk} = \sum_{i \in \mathcal{N}} x_{jik} \quad \forall j \in \mathcal{N} \setminus \{0, n+1\}, \; \forall k \in \mathcal{V}$$

### 6.2 Request Service

Each request is either served by exactly one vehicle or left unserved:

$$\sum_{k \in \mathcal{V}} y_{rk} + z_r = 1 \quad \forall r \in \mathcal{R}$$

If request $r$ is assigned to vehicle $k$, the vehicle must visit both the pickup and dropoff nodes:

$$y_{rk} \leq \sum_{j \in \mathcal{N}} x_{p_r, j, k} \quad \forall r \in \mathcal{R}, \; k \in \mathcal{V}$$

$$y_{rk} \leq \sum_{j \in \mathcal{N}} x_{q_r, j, k} \quad \forall r \in \mathcal{R}, \; k \in \mathcal{V}$$

### 6.3 Pickup Before Dropoff (Precedence)

$$t_{p_r, k} + \tau_{p_r, q_r}(t_{p_r,k}) \leq t_{q_r, k} \quad \forall r \in \mathcal{R}, \; k \in \mathcal{V} \text{ s.t. } y_{rk} = 1$$

### 6.4 Time Window Feasibility

$$a_r \leq t_{p_r, k} \quad \forall r \in \mathcal{R}, \; k \in \mathcal{V}$$

### 6.5 Temporal Consistency (Sub-Tour Elimination via MTZ)

$$t_{ik} + \tau_{ij}(t_{ik}) - T_{\max}(1 - x_{ijk}) \leq t_{jk} \quad \forall (i,j) \in \mathcal{E}, \; k \in \mathcal{V}$$

### 6.6 Vehicle Capacity

$$l_{ik} + w_r \cdot y_{rk} \leq Q_k \quad \forall i = p_r, \; r \in \mathcal{R}, \; k \in \mathcal{V}$$

$$0 \leq l_{ik} \leq Q_k \quad \forall i \in \mathcal{N}, \; k \in \mathcal{V}$$

### 6.7 Battery State-of-Charge

Battery decreases with travel and increases at charging stations:

$$b_{jk} \leq b_{ik} - e_k \cdot d_{ij} \cdot x_{ijk} + c_{ik} + B_k^{\max}(1 - x_{ijk}) \quad \forall (i,j) \in \mathcal{E}, \; k \in \mathcal{V}$$

$$B_k^{\min} \leq b_{ik} \leq B_k^{\max} \quad \forall i \in \mathcal{N}, \; k \in \mathcal{V}$$

Charging only occurs at designated stations:

$$c_{ik} = 0 \quad \forall i \notin \mathcal{S}, \; k \in \mathcal{V}$$

$$c_{ik} \leq B_k^{\max} - b_{ik} \quad \forall i \in \mathcal{S}, \; k \in \mathcal{V}$$

### 6.8 Initial Conditions

$$b_{0k} = B_k^{\max} \quad \forall k \in \mathcal{V}$$
$$l_{0k} = 0 \quad \forall k \in \mathcal{V}$$
$$t_{0k} = 0 \quad \forall k \in \mathcal{V}$$

---

## 7. Solution Methodology Comparison

This project implements and benchmarks multiple solution approaches to the E-DVRPTW:

| Approach | Method | Time Complexity | Solution Quality | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Exact** | Google OR-Tools CP-SAT (rolling-horizon) | Exponential (NP-hard) | Optimal within horizon | 10–500 ms |
| **Greedy Heuristic** | Weighted urgency scoring | $O(|\mathcal{R}| \cdot |\mathcal{V}|)$ | Good but myopic | < 0.1 ms |
| **Nearest Vehicle** | Euclidean distance matching | $O(|\mathcal{R}| \cdot |\mathcal{V}|)$ | Baseline | < 0.1 ms |
| **Deep RL (Maskable PPO)** | Neural network policy prior | $O(1)$ forward pass | Learned long-horizon | < 1 ms |
| **Hybrid PPO + MCTS** | Policy-guided tree search | $O(s \cdot d)$ simulations | Best overall | 5–45 ms |

Where $s$ = number of MCTS simulations and $d$ = rollout depth.

### 7.1 Key Insight: Learning-Augmented Optimization

The hybrid approach bridges the gap between exact methods and fast heuristics:

1. **PPO Policy Prior** filters the action space from $O(|\mathcal{R}| \times |\mathcal{V}|)$ combinatorial actions down to $K$ high-probability candidates.
2. **MCTS Forward Search** evaluates the downstream impact of each candidate by simulating future states on a cloned environment.
3. **Latency Budget Controller** guarantees real-time compliance ($<45$ ms) by falling back to the PPO action if tree search exceeds the time budget.

---

## 8. Multi-Objective Reward Function

The RL agent's reward function maps the mathematical objective to a scalar training signal:

$$R_t = \alpha \cdot N_{\text{deliv}} + \zeta \cdot U_{\text{fleet}} - \beta \cdot D_{\text{km}} - \gamma \cdot F_{\text{fuel}} - \delta \cdot N_{\text{SLA}} - \epsilon \cdot t_{\text{idle}} - \eta \cdot N_{\text{exp}} - \theta \cdot E_{\text{kWh}} - \kappa \cdot N_{\text{low\_batt}}$$

| Symbol | Weight | Component |
| :--- | :---: | :--- |
| $\alpha$ | 10.0 | Successful delivery reward |
| $\zeta$ | 1.0 | Fleet utilization bonus |
| $\beta$ | 0.1 | Travel distance penalty (per km) |
| $\gamma$ | 0.5 | Fuel/energy consumption penalty |
| $\delta$ | 20.0 | SLA violation penalty |
| $\epsilon$ | 0.05 | Idle time penalty (per minute) |
| $\eta$ | 15.0 | Request expiry penalty |
| $\theta$ | 0.3 | Carbon emissions penalty (per kWh) |
| $\kappa$ | 5.0 | Low battery penalty (vehicles below SoC threshold) |

Running z-score normalization stabilizes training:

$$\hat{R}_t = \text{clip}\left(\frac{R_t - \mu_R}{\sigma_R + 10^{-8}}, -10.0, 10.0\right)$$

---

## 9. References

1. Toth, P., & Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications*. MOS-SIAM Series on Optimization.
2. Schneider, M., Stenger, A., & Goeke, D. (2014). The electric vehicle-routing problem with time windows and recharging stations. *Transportation Science*, 48(4), 500–520.
3. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, learn to solve routing problems! *ICLR 2019*.
4. Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. *arXiv:1712.01815*.
5. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
