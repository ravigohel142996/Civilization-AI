# 🌍 Civilization AI — Autonomous World Simulation Engine

A **production-quality, multi-agent AI simulation platform** that models a world of AI-controlled nations evolving over time across economic, geopolitical, and technological dimensions.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **World Generator** | Procedurally generates 5–35 synthetic countries with realistic, correlated attributes |
| **GDP Predictor** | RandomForest regression model predicts future GDP based on technology, trade & resources |
| **Population Model** | Ridge regression estimates per-country annual population growth rates |
| **Conflict Predictor** | GradientBoosting classifier assesses pairwise geopolitical conflict probability |
| **Multi-Agent System** | Each country autonomously selects utility-maximising actions each round |
| **Trade Network** | NetworkX-powered directed graph tracking bilateral trade volumes and dependencies |
| **Simulation Engine** | Orchestrates all sub-systems across configurable time steps |
| **Interactive Dashboard** | Enterprise-style Streamlit UI with Plotly charts and live controls |

---

## 🏗️ Architecture

```
civilization-ai/
├── app.py                          # Streamlit entry point
├── config.py                       # Centralised configuration (no magic numbers)
│
├── data/
│   └── world_generator.py          # Synthetic world / country generation
│
├── models/
│   ├── gdp_predictor.py            # RandomForestRegressor GDP model
│   ├── population_growth_model.py  # Ridge regression population model
│   └── conflict_predictor.py       # GradientBoosting conflict classifier
│
├── simulation/
│   ├── civilization_engine.py      # Core simulation orchestrator
│   ├── agent_decision_system.py    # Multi-agent utility-maximisation
│   └── trade_network.py            # NetworkX trade graph
│
├── analytics/
│   └── global_metrics.py           # Aggregate KPIs & time-series derivation
│
├── ui/
│   ├── dashboard.py                # Section renderers (composed pages)
│   ├── charts.py                   # Plotly chart builders
│   ├── controls.py                 # Sidebar controls
│   └── world_map.py                # Scatter-geo world map
│
├── utils/
│   └── helpers.py                  # Formatting, normalisation utilities
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web dashboard
- **Scikit-learn** — ML models (RandomForest, Ridge, GradientBoosting)
- **Pandas / NumPy** — data manipulation
- **Plotly** — interactive charts and maps
- **NetworkX** — trade-network graph analytics

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Configure & simulate

1. Use the **sidebar sliders** to set the number of countries, resource level, technology growth rate, climate stress, and simulation rounds.
2. Click **🚀 Run Simulation**.
3. Navigate between dashboard sections using the top radio buttons.

---

## 📊 Dashboard Sections

| Section | Contents |
|---|---|
| **Global Overview** | World population, global GDP, stability & conflict KPIs; aggregate trend chart |
| **Economy Analytics** | Per-country GDP growth, population trends, technology index; top economies bar |
| **Trade Network** | Interactive trade graph, top hubs by PageRank centrality, edge table |
| **Conflict Monitor** | Conflict risk time-series, per-country risk bar chart, risk table |
| **World Map** | Scatter-geo map colour-coded by any metric (GDP, population, stability…) |
| **Country Comparison** | Radar chart for multi-country attribute comparison; full data table |

---

## 🔮 Future Improvements

- **Reinforcement learning agents** — replace utility-maximisation with trained RL policies (e.g., PPO via Stable-Baselines3)
- **Historical calibration** — initialise world state from real World Bank data
- **Diplomacy & alliances** — model treaty formation and coalition dynamics
- **Resource depletion** — finite resource mechanics with extraction curves
- **Climate model integration** — link climate risk to actual emissions trajectories
- **Persistent scenarios** — save/load simulation states and replay histories
- **Multi-user collaboration** — allow multiple users to control different countries in real time
