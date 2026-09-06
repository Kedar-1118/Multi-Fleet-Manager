# Multi-Fleet Manager: Dynamic Fleet Dispatch & Vehicle Routing Optimization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.0-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.1.0-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-164%20passed-brightgreen.svg)](dynamic-fleet-routing/tests/)

A production-grade, hybrid optimization engine combining **Deep Reinforcement Learning (Maskable PPO)**, **Monte Carlo Tree Search (MCTS)**, and **Operations Research (Google OR-Tools CP-SAT)** for dynamic urban logistics, real-time vehicle routing, and high-density fleet dispatching under stochastic traffic and strict SLA deadlines.

---

## 1. What Problem This Project Solves

### The Real-World Challenge: Dynamic Urban Last-Mile Logistics
In modern metropolitan courier, on-demand grocery, food delivery, and ride-hailing networks, dispatchers must coordinate vehicle fleets under extreme uncertainty:

1. **Dynamic, Stochastically Arriving Demand**: Thousands of customer requests arrive dynamically throughout the day following non-homogeneous Poisson processes rather than static, known schedules.
2. **Volatile Traffic & Bottlenecks**: Urban road speeds fluctuate wildly due to morning (08:00–10:00) and evening (17:00–20:00) rush-hour congestion, plus random localized traffic incidents.
3. **Strict Service Level Agreements (SLAs)**: Every package or passenger has strict deadlines. Late arrivals incur heavy penalties, SLA breaches, and customer churn.
4. **Physical Vehicle Constraints**: Vehicles operate with finite payload capacities, discrete operational states (`IDLE`, `MOVING_TO_PICKUP`, `MOVING_TO_DROPOFF`, `SERVICING`), and continuous fuel consumption.

### The Computational Dilemma
Solving the Dynamic Vehicle Routing Problem with Time Windows (DVRPTW) presents a classic tradeoff between computational tractability and decision quality:

* **Traditional Operations Research (OR) Solvers** (e.g., Integer Programming, Branch-and-Cut, CP-SAT):
  * Guarantee high solution optimality for static snapshots.
  * **The Problem**: Finding exact solutions is **NP-hard**. Under high request density and continuous arrivals, re-solving full combinatorial route assignments causes exponential latency scaling, making sub-second real-time dispatching infeasible.
* **Pure Deep Reinforcement Learning (RL)**:
  * Offers sub-millisecond forward-pass execution ($<1\,\text{ms}$ inference).
  * **The Problem**: Pure RL policies without lookahead planning suffer from high variance and struggle with long-horizon sequential rollouts in out-of-distribution traffic conditions.

### The Solution: A Hybrid Real-Time Dispatch System
This project closes this gap by combining **Maskable PPO** with **Bounded MCTS forward search** protected by a **hardware latency budget controller**:
* **Policy Prior**: Deep RL filters the massive combinatorial action space down to high-probability candidate assignments in $<1\,\text{ms}$.
* **Forward Tree Search**: Bounded MCTS simulates multi-step future rollouts from cloned states to verify assignment quality before committing.
* **SLA Safety Fallback**: If tree search threatens a hard latency deadline ($<45\,\text{ms}$), the engine automatically falls back to the top PPO policy decision.

---

## 2. Technology Stack

| Domain | Technology / Framework | Version | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Simulation Core** | **Gymnasium** | `^0.29.0` | Standardized environment interface (`DynamicFleetEnv`) modeling continuous dynamic states. |
| | **NetworkX** | `^3.2` | Road network graph modeling, intersection topologies, edge weights, and Dijkstra shortest path caching. |
| | **Python `heapq`** | Built-in | Discrete-event priority queue driving continuous-time simulation without wasteful fixed-interval ticks. |
| **Deep Reinforcement Learning** | **PyTorch** | `^2.1.0` | Neural network policy layers, GPU/CPU tensor computations, and TorchScript model serialization. |
| | **Stable-Baselines3** | `^2.1.0` | Production RL algorithms, vector environments, and training callbacks. |
| | **`sb3-contrib`** | `^2.1.0` | `MaskablePPO` implementation supporting action validity masking. |
| **Search & Optimization** | **Custom MCTS Engine** | Custom | State-cloning Monte Carlo Tree Search with UCB1 exploration and rollout evaluation. |
| | **Google OR-Tools** | `^9.8` | CP-SAT constraint programming solver used for OR benchmark comparisons. |
| **Serving & Deployment** | **FastAPI** | `^0.104.0` | High-performance asynchronous REST API (`/dispatch`, `/simulate`, `/health`, `/metrics`). |
| | **Uvicorn** | `^0.24.0` | Lightning-fast ASGI production web server. |
| | **Pydantic** | `^2.5.0` | Strict data validation schemas for fleet states, dispatch inputs, and responses. |
| | **Docker & Docker Compose** | Multi-stage | Containerization of API server and MLflow tracking service. |
| **Experimentation & Tuning** | **MLflow** | `^2.8.0` | Experiment tracking, hyperparameter logging, and model artifact storage. |
| | **Ray Tune** | `^2.8.0` | Distributed hyperparameter tuning with ASHA scheduler. |
| **Testing & Quality** | **Pytest & Pytest-Cov** | `^8.0` | Comprehensive 164-test suite validating environment invariants, baselines, and serving. |

---

## 3. How the System Works (Architecture & Algorithms)

```
                       [ Incoming Requests & Fleet States ]
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  FastAPI Serving Layer /api   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │         HybridPlanner         │
                       └───────┬───────────────┬───────┘
                               │               │
       ┌───────────────────────┘               └───────────────────────┐
       ▼                                                               ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│         Maskable PPO          │                       │     OR-Tools & Baselines      │
│  Valid Action Masking (<1ms)  │                       │ Nearest Vehicle, Greedy, CP-SAT│
└──────────────┬────────────────┘                       └───────────────────────────────┘
               │ Action Priors P(s, a)
               ▼
┌───────────────────────────────┐
│    Latency Budget Check       │
└──────┬─────────────────┬──────┘
       │                 │
 [< 5ms Left]     [Sufficient Time]
       ▼                 ▼
Fallback to PPO   Bounded MCTS Tree Search
Action Directly   (Clone State -> Forward Rollouts -> UCB1)
       │                 │
       └────────┬────────┘
                ▼
      [ Execute Dispatch ]
                │
                ▼
┌───────────────────────────────┐
│     DynamicFleetEnv (Gym)     │
│   Discrete-Event Queue (heapq)│
│   NetworkX Road Graph         │
│   Time-Varying Traffic Model  │
└───────────────────────────────┘
```

### 3.1 Discrete-Event Simulation (`src/environment/`)
Rather than stepping in arbitrary 1-second fixed time intervals, `DynamicFleetEnv` advances continuously using an event priority queue (`heapq`):
* **Events**: `NEW_REQUEST`, `ASSIGNED`, `VEHICLE_ARRIVAL`, `PICKUP_COMPLETE`, `DROPOFF_COMPLETE`, and `REQUEST_EXPIRATION`.
* **Traffic Model**: Road edge traversal times incorporate dynamic time-of-day multipliers:
  $$T(u, v, t) = \frac{d(u, v)}{v_{\text{base}}} \cdot M(t, u, v) \cdot 60 \quad (\text{minutes})$$
  * Multipliers: `LOW_TRAFFIC` ($0.8\times$), `NORMAL` ($1.0\times$), `RUSH_HOUR` ($2.0\times$–$2.5\times$), plus stochastic localized incident delays ($+1.5\times$ for 15–45 min).

### 3.2 Action Space & Invalid Action Masking (`src/agents/`)
With $V$ vehicles and $K$ top pending requests, the discrete action space is:
$$\text{action} = \text{request\_idx} \times V + \text{vehicle\_idx}$$
plus a terminal `NO-OP` (wait) action.
* **Dynamic Masking**: Actions that violate physical constraints are strictly masked to zero probability:
  * Vehicles currently in transit or servicing (`MOVING_TO_PICKUP`, `MOVING_TO_DROPOFF`, `SERVICING`).
  * Requests that are already assigned, delivered, or expired.
  * Requests whose payload size exceeds the vehicle's remaining capacity.

### 3.3 Monte Carlo Tree Search with Policy Priors (`src/planning/`)
When execution time allows, `MCTSPlanner` creates a deep clone of the simulation state (`clone_state()`) and performs lookahead planning using **Polynomial Upper Confidence Trees (PUCT / UCB1)**:
$$\text{UCB1}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
* $P(s, a)$ is the prior probability distribution produced by the Maskable PPO network, ensuring the tree prioritizes high-quality decisions instead of exploring randomly.
* Zero side-effects: cloned states prevent simulation corruption during search.

### 3.4 Multi-Objective Reward Function (`src/environment/reward.py`)
To align the agent with real-world business KPIs, the reward balances service quality, throughput, and costs:
$$R_t = \alpha \cdot N_{\text{deliv}} + \zeta \cdot U_{\text{fleet}} - \beta \cdot D_{\text{km}} - \gamma \cdot F_{\text{fuel}} - \delta \cdot N_{\text{SLA}} - \epsilon \cdot t_{\text{idle}} - \eta \cdot N_{\text{exp}}$$
Running z-score normalization clips extreme variance:
$$\hat{R}_t = \text{clip}\left(\frac{R_t - \mu_R}{\sigma_R + 10^{-8}}, -10.0, 10.0\right)$$

---

## 4. Measured Performance Benchmarks

### 4.1 Inference Latency Benchmark (1,000 Iterations)
| Method | Mean Latency | Median (P50) | P95 Latency | P99 Latency | SLA Compliant (<45ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nearest Vehicle** | 0.056 ms | 0.009 ms | 0.300 ms | 0.647 ms | 100.0% |
| **Greedy Dispatch** | 0.017 ms | 0.014 ms | 0.029 ms | 0.043 ms | 100.0% |
| **MCTS (10 simulations)** | 6.987 ms | 6.858 ms | 8.523 ms | 9.942 ms | 100.0% |
| **MCTS (50 simulations)** | 75.504 ms | 32.332 ms | 192.176 ms | 225.553 ms | Fallback Triggered |

### 4.2 Full 24-Hour Simulation Episode Results
| Dispatch Method | Completion Rate | SLA Compliance | Avg Turnaround | Total Distance | Total Fuel | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nearest Vehicle** | 78.7% | **84.5%** | **48.05 min** | 1,418 km | 113.5 units | 0.297 ms |
| **Greedy Dispatch** | 76.1% | 70.8% | 54.96 min | 1,597 km | 127.8 units | 0.314 ms |
| **OR-Tools CP-SAT** | 78.8% | 76.6% | 50.09 min | 1,447 km | 115.8 units | 16.68 ms |
| **PPO Agent** | 65.8% | 83.3% | 53.38 min | 1,616 km | 129.3 units | 0.574 ms |
| **MCTS (10 sims)** | **87.2%** | 74.2% | 52.80 min | 145 km* | 11.6 units* | 12.64 ms |

---

## 5. How to Test and Run the Project

### 5.1 Prerequisites & Setup
Navigate to the module directory:
```powershell
cd "k:\Projects\Multi-Fleet Manager\dynamic-fleet-routing"
pip install -e ".[all]"
```

### 5.2 Automated Testing (`pytest`)
The project includes **164 automated unit, integration, and invariant tests**:

* **Run all tests**:
  ```powershell
  pytest tests/ -v --tb=short
  ```
* **Run with test coverage report**:
  ```powershell
  pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
  ```
* **Run specific test suites**:
  ```powershell
  # Environment & Dynamic Requests
  pytest tests/test_environment.py tests/test_requests.py -v

  # Dispatch Baselines (Nearest, Greedy, OR-Tools)
  pytest tests/test_baselines.py -v

  # MCTS & Hybrid Planner
  pytest tests/test_mcts.py tests/test_hybrid_planner.py -v

  # Physical Invariants & Action Masking
  pytest tests/test_invariants.py tests/test_action_validity.py -v

  # Serving & FastAPI Endpoints
  pytest tests/test_serving.py tests/test_inference.py -v
  ```

### 5.3 Benchmarks and Simulation Experiments
* **Evaluate All Dispatch Methods**:
  ```powershell
  python -m src.training.evaluate --config configs/base.yaml
  ```
* **Run Latency Benchmark**:
  ```powershell
  python scripts/benchmark_latency.py
  ```
* **Run Ablation Experiments**:
  ```powershell
  python scripts/run_experiment.py
  ```

### 5.4 Starting and Testing the REST API

#### Start the Server:
```powershell
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

#### Test Endpoints (PowerShell):
> **Note on Windows PowerShell**: Use `curl.exe` (or `Invoke-RestMethod`) because standard `curl` is aliased to `Invoke-WebRequest`.

* **Health Check**:
  ```powershell
  curl.exe http://localhost:8000/health
  ```
  *Response:* `{"status":"healthy","model_loaded":false,"version":"1.0.0"}`

* **Real-Time Dispatch Request**:
  ```powershell
  curl.exe -X POST http://localhost:8000/dispatch --json '{\"vehicles\": [{\"vehicle_id\": 0, \"current_location\": 12, \"capacity\": 10, \"current_load\": 0, \"status\": \"IDLE\", \"fuel_remaining\": 100.0}, {\"vehicle_id\": 1, \"current_location\": 34, \"capacity\": 10, \"current_load\": 2, \"status\": \"IDLE\", \"fuel_remaining\": 88.5}], \"pending_requests\": [{\"request_id\": 101, \"pickup_location\": 12, \"dropoff_location\": 45, \"deadline_minutes\": 45.0, \"priority\": 2, \"package_size\": 1}], \"traffic_state\": \"NORMAL_TRAFFIC\", \"current_time\": 510.0, \"method\": \"auto\"}'
  ```

* **Interactive Swagger UI**:
  Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to inspect schemas and run requests interactively.

---

## 6. Directory Structure

```text
Multi-Fleet Manager/
├── README.md                      # Primary project documentation (this file)
├── PROJECT_STATUS.md              # Implementation status, test results & memory reference
└── dynamic-fleet-routing/         # Core application package
    ├── pyproject.toml             # Package dependencies and configuration
    ├── Dockerfile                 # Container image specification
    ├── docker-compose.yml         # API and MLflow orchestration
    ├── Makefile                   # Quick developer shortcuts
    ├── configs/                   # Simulation, PPO, MCTS, and tuning YAML configs
    ├── src/
    │   ├── environment/           # Discrete-event simulator, road network & traffic
    │   ├── agents/                # Maskable PPO policy & action masking
    │   ├── planning/              # State-cloned MCTS & latency-budget HybridPlanner
    │   ├── baselines/             # Greedy, Nearest Vehicle & Google OR-Tools CP-SAT
    │   ├── training/              # PPO training, evaluation & Ray Tune
    │   ├── serving/               # FastAPI REST service & TorchScript inference
    │   └── utils/                 # Structured logging, metrics, seed management
    ├── tests/                     # 164 unit, invariant, and integration tests
    ├── scripts/                   # Benchmarks, ablation runners & plot generation
    └── artifacts/                 # Saved models, metrics, plots & datasets
```

---

## 7. License
This project is licensed under the [MIT License](dynamic-fleet-routing/LICENSE).
