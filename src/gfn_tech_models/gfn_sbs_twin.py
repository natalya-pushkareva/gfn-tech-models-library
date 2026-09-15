import streamlit as st
import networkx as nx
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')

# ====================== КОНФИГУРАЦИЯ ======================
REALISTIC_2026_PARAMS = {
    'sbs_share_USA': 0.38,
    'sbs_share_China': 0.32,
    'sbs_share_Russia': 0.55,
    'ponzi_growth_rate': 0.015,
    'margin_call_severity': 0.20,
    'intervention_effectiveness': 0.60,
    'shock_probability': 0.025,
    'russia_sanctions_pressure': 0.55,
    'initial_cci': 38,
    'initial_haircut': 0.28,
    'initial_confidence': 55.0
}

BASE_PARAMS = {
    'sbs_share_USA': 0.35,
    'sbs_share_China': 0.28,
    'sbs_share_Russia': 0.38,
    'ponzi_growth_rate': 0.010,
    'margin_call_severity': 0.15,
    'intervention_effectiveness': 0.65,
    'shock_probability': 0.012,
    'russia_sanctions_pressure': 0.30
}


class AgentStatus(Enum):
    CORE_SAVED = "core_saved"
    CORE_ABANDONED = "core_abandoned"
    PERIPHERY_SURVIVING = "periphery_surviving"
    PERIPHERY_SUBSIDIZING = "periphery_subsidizing"
    PERIPHERY_DEFAULTED = "periphery_defaulted"
    EXILED = "exiled"


# ====================== АГЕНТ ======================
class Spider:
    def __init__(self, aid, spider_type, strength, params: Dict):
        self.id = aid
        self.type = spider_type
        self.geo_name = None
        self.base_strength = strength
        self.risk_appetite = 0.35
        self.ponzi_level = 0.0
        
        self.sbs_share = 0.0
        self.haircut = 0.15
        self.collateral_ratio = 1.0
        self.margin_call_history = 0
        self.bank_fail = False
        
        self.status = AgentStatus.PERIPHERY_SURVIVING
        self.subsidy_paid = 0.0
        self.bailout_received = 0.0
        self.sanctions_pressure = 0.0
        self.is_exiled = False
        self.params = params

    def set_geo_attributes(self, geo_name, sbs_share):
        self.geo_name = geo_name
        self.sbs_share = sbs_share

    def update(self, contagion_factor=0.0):
        growth_rate = self.params.get('ponzi_growth_rate', 0.010)
        self.risk_appetite = min(1.0, self.risk_appetite + growth_rate + contagion_factor * 0.02)
        
        if self.sanctions_pressure > 0:
            self.risk_appetite *= (1 + self.sanctions_pressure * 0.08)
        
        self.ponzi_level = max(0.0, self.risk_appetite - 0.52)

    def apply_margin_call(self, severity=0.1):
        self.margin_call_history += 1
        self.collateral_ratio = max(0.15, self.collateral_ratio * (1 - severity))
        self.ponzi_level = min(0.95, self.ponzi_level + 0.02 + severity * 0.03)

    def apply_subsidy(self, amount: float):
        self.subsidy_paid += amount
        self.ponzi_level = min(0.85, self.ponzi_level + amount * 0.025)
        self.haircut = min(0.55, self.haircut + amount * 0.03)
        if self.ponzi_level > 0.78:
            self.status = AgentStatus.PERIPHERY_DEFAULTED

    def receive_bailout(self, amount: float):
        self.bailout_received += amount
        self.ponzi_level = max(0.35, self.ponzi_level - amount * 0.08)
        self.haircut = max(0.20, self.haircut - amount * 0.06)
        self.status = AgentStatus.CORE_SAVED

    def apply_sanctions(self, pressure: float):
        self.sanctions_pressure = pressure
        self.sbs_share = min(0.85, self.sbs_share + pressure * 0.08)

    def exile(self):
        self.is_exiled = True
        self.status = AgentStatus.EXILED
        self.ponzi_level = 0.90


# ====================== ГИБРИДНАЯ МОДЕЛЬ ======================
class HybridGFNSBS:
    def __init__(self, n_agents=90, seed=42, params=None, disable_bailout=False):
        random.seed(seed)
        np.random.seed(seed)
        
        self.USA_ID = 0
        self.CHINA_ID = 1
        self.RUSSIA_ID = 2
        
        self.params = params if params else BASE_PARAMS.copy()
        self.disable_bailout = disable_bailout
        
        self.agents = {}
        self.layers = {name: nx.Graph(name=name) for name in ['financial', 'legal']}
        
        self.sbs_crash_triggered = False
        self.global_crisis_triggered = False
        self.intervention_triggered = False
        self.bailout_active = False
        self.russia_exiled = False
        
        self.haircut_history = []
        self.ponzi_history = []
        self.cci_history = []
        
        self.create_agents(n_agents)
        self.initialize_layers()
        
        self.total_liquidity = 100.0
        self.total_collateral = 85.0
        self.systemic_confidence = 90.0
        self.bailout_cost = 0.0
        self.subsidy_collected = 0.0

    def create_agents(self, n):
        for i in range(n):
            if i < 7:
                typ, s = 'core', random.uniform(0.85, 1.0)
            elif i < 22:
                typ, s = 'large', random.uniform(0.55, 0.85)
            else:
                typ, s = 'peripheral', random.uniform(0.10, 0.55)
            
            agent = Spider(i, typ, s, self.params)
            
            if typ == 'core':
                agent.sbs_share = random.uniform(0.20, 0.40)
                agent.haircut = random.uniform(0.10, 0.20)
            elif typ == 'large':
                agent.sbs_share = random.uniform(0.35, 0.55)
                agent.haircut = random.uniform(0.18, 0.28)
            else:
                agent.sbs_share = random.uniform(0.55, 0.80)
                agent.haircut = random.uniform(0.25, 0.40)
            
            self.agents[i] = agent
        
        self.agents[self.USA_ID].set_geo_attributes("USA", self.params['sbs_share_USA'])
        self.agents[self.CHINA_ID].set_geo_attributes("China", self.params['sbs_share_China'])
        self.agents[self.RUSSIA_ID].set_geo_attributes("Russia", self.params['sbs_share_Russia'])

    def initialize_layers(self):
        for layer_name, G in self.layers.items():
            for aid, spider in self.agents.items():
                G.add_node(aid, 
                          type=spider.type, 
                          geo=spider.geo_name,
                          strength=spider.base_strength)
            
            core_nodes = [aid for aid, s in self.agents.items() if s.type == 'core']
            peripheral_nodes = [aid for aid, s in self.agents.items() if s.type != 'core']
            
            for i, j in combinations(core_nodes, 2):
                if random.random() < 0.7:
                    G.add_edge(i, j, weight=random.uniform(0.7, 1.0))
            
            for c in core_nodes:
                for p in peripheral_nodes:
                    if random.random() < 0.3:
                        G.add_edge(c, p, weight=random.uniform(0.3, 0.7))
            
            for i, j in combinations(peripheral_nodes, 2):
                if random.random() < 0.1:
                    G.add_edge(i, j, weight=random.uniform(0.2, 0.5))

    def get_geo_name(self, aid):
        return self.agents[aid].geo_name if aid in self.agents else None

    def compute_cci(self) -> float:
        active_agents = [a for a in self.agents.values() if not a.is_exiled]
        if not active_agents:
            return 100.0
        
        avg_ponzi = np.mean([a.ponzi_level for a in active_agents])
        avg_haircut = np.mean([a.haircut for a in active_agents if a.sbs_share > 0.2]) if any(a.sbs_share > 0.2 for a in active_agents) else 0.15
        
        ponzi_norm = min(1.0, avg_ponzi / 0.60)
        haircut_norm = min(1.0, avg_haircut / 0.50)
        confidence_norm = (100 - self.systemic_confidence) / 100.0
        
        cci = (0.45 * ponzi_norm + 0.35 * haircut_norm + 0.20 * confidence_norm) * 100
        return cci

    def warm_up(self, steps=40):
        for _ in range(steps):
            self.step()
        return self

    def calibrate_to_2026_reality(self):
        """Калибровка на реалистичное состояние 2026 года"""
        params = REALISTIC_2026_PARAMS
        
        self.params['sbs_share_USA'] = params['sbs_share_USA']
        self.params['sbs_share_China'] = params['sbs_share_China']
        self.params['sbs_share_Russia'] = params['sbs_share_Russia']
        self.params['ponzi_growth_rate'] = params['ponzi_growth_rate']
        self.params['shock_probability'] = params['shock_probability']
        
        for agent in self.agents.values():
            if agent.geo_name == "USA":
                agent.sbs_share = params['sbs_share_USA']
                agent.ponzi_level = 0.28
                agent.haircut = 0.18
            elif agent.geo_name == "China":
                agent.sbs_share = params['sbs_share_China']
                agent.ponzi_level = 0.32
                agent.haircut = 0.22
            elif agent.geo_name == "Russia":
                agent.sbs_share = params['sbs_share_Russia']
                agent.ponzi_level = 0.48
                agent.haircut = 0.35
                if params['russia_sanctions_pressure'] > 0.4:
                    agent.apply_sanctions(params['russia_sanctions_pressure'])
            else:
                if agent.type == 'peripheral':
                    agent.ponzi_level = random.uniform(0.35, 0.55)
                    agent.haircut = random.uniform(0.28, 0.42)
                elif agent.type == 'large':
                    agent.ponzi_level = random.uniform(0.25, 0.40)
                    agent.haircut = random.uniform(0.20, 0.30)
        
        self.systemic_confidence = params['initial_confidence']
        self.total_liquidity = 65.0
        self.total_collateral = 55.0
        
        for _ in range(5):
            self.cci_history.append(params['initial_cci'])
        
        return self

    def apply_external_shock(self):
        shock_prob = self.params.get('shock_probability', 0.015)
        if random.random() < shock_prob:
            severity = random.uniform(0.05, 0.25)
            for agent in self.agents.values():
                if not agent.is_exiled and random.random() < 0.3:
                    agent.ponzi_level = min(0.90, agent.ponzi_level + severity)

    def apply_sbs_spiral(self):
        active_agents = [a for a in self.agents.values() if not a.is_exiled]
        if not active_agents:
            return
        
        avg_haircut = np.mean([a.haircut for a in active_agents if a.sbs_share > 0.2]) if any(a.sbs_share > 0.2 for a in active_agents) else 0.15
        avg_ponzi = np.mean([a.ponzi_level for a in active_agents])
        
        self.haircut_history.append(avg_haircut)
        self.ponzi_history.append(avg_ponzi)
        
        if avg_ponzi > 0.38:
            for agent in active_agents:
                if agent.sbs_share > 0.3:
                    agent.haircut = min(0.65, agent.haircut * (1 + 0.04))
                    if agent.haircut > 0.35:
                        agent.apply_margin_call(severity=0.10)
        
        if avg_haircut > 0.42 and not self.sbs_crash_triggered:
            self.sbs_crash_triggered = True
            for agent in active_agents:
                if agent.sbs_share > 0.4:
                    agent.ponzi_level = min(0.85, agent.ponzi_level + 0.05)

    def apply_regulatory_intervention(self):
        avg_ponzi = np.mean([a.ponzi_level for a in self.agents.values() if not a.is_exiled])
        
        if avg_ponzi > 0.55 and not self.intervention_triggered:
            self.intervention_triggered = True
            effectiveness = self.params.get('intervention_effectiveness', 0.65)
            
            for agent in self.agents.values():
                if not agent.is_exiled and agent.type == 'core':
                    agent.ponzi_level = max(0.35, agent.ponzi_level - 0.10 * effectiveness)
                    agent.haircut = max(0.20, agent.haircut - 0.08 * effectiveness)

    def apply_selective_bailout(self):
        if self.disable_bailout:
            return
        
        avg_ponzi = np.mean([a.ponzi_level for a in self.agents.values() if not a.is_exiled])
        avg_haircut = np.mean([a.haircut for a in self.agents.values() if a.sbs_share > 0.2]) if any(a.sbs_share > 0.2 for a in self.agents.values()) else 0.2
        
        if (avg_ponzi > 0.50 or avg_haircut > 0.38 or self.sbs_crash_triggered) and not self.bailout_active:
            self.bailout_active = True
            
            to_save = [a for a in self.agents.values() if a.geo_name in ["USA", "China"]]
            to_subsidize = [a for a in self.agents.values() 
                           if a.geo_name not in ["USA", "China"] and not a.is_exiled and a.type != 'core']
            
            bailout_amount = 6.0
            total_cost = len(to_save) * bailout_amount
            
            if to_subsidize:
                subsidy = total_cost / len(to_subsidize)
            else:
                subsidy = 0
            
            for agent in to_save:
                agent.receive_bailout(bailout_amount)
            
            for agent in to_subsidize:
                agent.apply_subsidy(subsidy)
            
            self.bailout_cost += total_cost
            self.subsidy_collected += len(to_subsidize) * subsidy
            
            russia = self.agents.get(self.RUSSIA_ID)
            if russia and not russia.is_exiled:
                sanctions = self.params.get('russia_sanctions_pressure', 0.30)
                russia.apply_sanctions(sanctions)
                
                if russia.ponzi_level > 0.70 and not self.russia_exiled:
                    self.russia_exiled = True
                    russia.exile()
                    for G in self.layers.values():
                        if russia.id in G:
                            G.remove_node(russia.id)

    def update_system_dynamics(self):
        avg_ponzi = np.mean([a.ponzi_level for a in self.agents.values() if not a.is_exiled])
        avg_haircut = np.mean([a.haircut for a in self.agents.values() if a.sbs_share > 0.2]) if any(a.sbs_share > 0.2 for a in self.agents.values()) else 0.15
        
        if avg_ponzi > 0.45 or avg_haircut > 0.35:
            self.systemic_confidence *= 0.98
            self.systemic_confidence = max(20, self.systemic_confidence)
        else:
            self.systemic_confidence = min(100, self.systemic_confidence * 1.005)
        
        self.total_liquidity = max(30, min(150, self.total_liquidity + (0.1 - avg_ponzi * 0.2)))

    def step(self):
        for agent in self.agents.values():
            if not agent.is_exiled:
                agent.update()
        
        self.apply_external_shock()
        self.apply_sbs_spiral()
        self.apply_selective_bailout()
        self.apply_regulatory_intervention()
        self.update_system_dynamics()
        
        cci = self.compute_cci()
        self.cci_history.append(cci)
        return cci

    def run_forecast(self, months: int):
        cci_trajectory = []
        for _ in range(months):
            cci = self.step()
            cci_trajectory.append(cci)
        return cci_trajectory

    def get_position_in_gfn(self, node_ids):
        results = {}
        G_fin = self.layers['financial']
        
        degree_centrality = nx.degree_centrality(G_fin)
        betweenness = nx.betweenness_centrality(G_fin, weight='weight')
        clustering = nx.clustering(G_fin, weight='weight')
        
        for nid in node_ids:
            if nid not in self.agents:
                continue
            agent = self.agents[nid]
            results[nid] = {
                'name': self.get_geo_name(nid) or f"Agent_{nid}",
                'type': agent.type,
                'strength': agent.base_strength,
                'ponzi': agent.ponzi_level,
                'degree_c': degree_centrality.get(nid, 0),
                'betweenness': betweenness.get(nid, 0),
                'clustering': clustering.get(nid, 0),
                'full_member': (agent.type == 'core' and degree_centrality.get(nid, 0) > 0.08),
                'sbs_share': agent.sbs_share,
                'haircut': agent.haircut,
                'collateral': agent.collateral_ratio,
                'is_exiled': agent.is_exiled,
                'status': agent.status.value
            }
        return results


# ====================== ВИЗУАЛИЗАЦИЯ СЕТИ ======================
def create_network_plot(G, sim, layer_name):
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    fig = go.Figure()
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], 
            mode='lines', 
            line=dict(width=edge[2]['weight']*1.5, color='lightgray'), 
            opacity=0.4,
            hoverinfo='none'
        ))
    
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    
    degree_centrality = nx.degree_centrality(G)
    
    for n in G.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        
        if n == sim.USA_ID:
            node_colors.append('#1f77b4')
            node_sizes.append(55)
        elif n == sim.CHINA_ID:
            node_colors.append('#d62728')
            node_sizes.append(52)
        elif n == sim.RUSSIA_ID:
            if sim.agents[n].is_exiled:
                node_colors.append('#808080')
            else:
                node_colors.append('#2ca02c')
            node_sizes.append(28)
        elif sim.agents[n].type == 'core':
            node_colors.append('#ff7f0e')
            node_sizes.append(sim.agents[n].base_strength * 45)
        elif sim.agents[n].type == 'large':
            node_colors.append('#ffbb78')
            node_sizes.append(sim.agents[n].base_strength * 35)
        else:
            node_colors.append('#d3d3d3')
            node_sizes.append(sim.agents[n].base_strength * 25)
        
        geo_tag = sim.get_geo_name(n)
        name_str = f"🌍 {geo_tag}" if geo_tag else f"Agent {n}"
        agent = sim.agents[n]
        node_texts.append(
            f"{name_str}<br>"
            f"Type: {agent.type}<br>"
            f"Strength: {agent.base_strength:.3f}<br>"
            f"Ponzi: {agent.ponzi_level:.3f}<br>"
            f"SBS share: {agent.sbs_share:.2f}<br>"
            f"Haircut: {agent.haircut:.3f}<br>"
            f"Exiled: {'⚠️ YES' if agent.is_exiled else 'No'}<br>"
            f"Centrality: {degree_centrality.get(n, 0):.3f}"
        )
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        mode='markers',
        marker=dict(
            size=node_sizes, 
            color=node_colors, 
            line=dict(width=1.5, color='black'),
            opacity=0.85
        ),
        text=node_texts,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"🌐 Слой: {layer_name.capitalize()} — США (🔵), Китай (🔴), РФ (🟢/⚪)",
        showlegend=False, 
        height=600, 
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig


# ====================== TWIN-SIMULATION ======================
def run_twin_simulation_realistic(n_agents, forecast_years, seed, n_simulations=10):
    forecast_months = forecast_years * 12
    years = list(range(2026, 2026 + forecast_years))
    
    all_with = []
    all_without = []
    
    progress_bar = st.progress(0)
    
    for i in range(n_simulations):
        sim_with = HybridGFNSBS(n_agents=n_agents, seed=seed + i, 
                                params=BASE_PARAMS.copy(), disable_bailout=False)
        sim_with.calibrate_to_2026_reality()
        sim_with.warm_up(10)
        
        traj_with = []
        for _ in range(forecast_months):
            cci = sim_with.step()
            traj_with.append(cci)
        all_with.append(traj_with)
        
        sim_without = HybridGFNSBS(n_agents=n_agents, seed=seed + i + 1000,
                                   params=BASE_PARAMS.copy(), disable_bailout=True)
        sim_without.calibrate_to_2026_reality()
        sim_without.warm_up(10)
        
        traj_without = []
        for _ in range(forecast_months):
            cci = sim_without.step()
            traj_without.append(cci)
        all_without.append(traj_without)
        
        progress_bar.progress((i + 1) / n_simulations)
    
    all_with = np.array(all_with)
    all_without = np.array(all_without)
    
    mean_with = np.mean(all_with, axis=0)
    std_with = np.std(all_with, axis=0)
    mean_without = np.mean(all_without, axis=0)
    std_without = np.std(all_without, axis=0)
    
    yearly_with = [mean_with[i*12] if i*12 < len(mean_with) else mean_with[-1] 
                   for i in range(forecast_years)]
    yearly_without = [mean_without[i*12] if i*12 < len(mean_without) else mean_without[-1] 
                      for i in range(forecast_years)]
    
    return {
        'years': years,
        'with': {'mean': yearly_with, 'std': [std_with[i*12] if i*12 < len(std_with) else std_with[-1] 
                                               for i in range(forecast_years)]},
        'without': {'mean': yearly_without, 'std': [std_without[i*12] if i*12 < len(std_without) else std_without[-1] 
                                                     for i in range(forecast_years)]},
        'initial_cci': REALISTIC_2026_PARAMS['initial_cci']
    }


# ====================== ОСНОВНОЙ ДАШБОРД ======================
st.set_page_config(page_title="GFN+SBS: Realistic 2026 Calibration", layout="wide")

st.title("🌐 GFN+SBS: Twin-Simulation Framework")
st.markdown("""
**Сравнение двух режимов с реалистичной калибровкой на 2026 год:**

- **Режим A (с вмешательством)** — Selective bailout + TBTF rescue
- **Режим B (без вмешательства)** — Pure market dynamics

**Россия (🟢)** — периферийный агент с высокой SBS-долей (55%), уязвимый к изгнанию
""")

# Сайдбар
with st.sidebar:
    st.header("⚙️ Параметры")
    n_agents = st.slider("Количество агентов", 60, 120, 90)
    forecast_years = st.slider("Горизонт прогноза (лет)", 3, 7, 5)
    seed = st.number_input("Random Seed", value=42, step=1)
    n_simulations = st.slider("Monte Carlo запусков", 5, 20, 10)
    
    st.markdown("---")
    run_twin_btn = st.button("🚀 Запустить Twin-Simulation", type="primary")
    warm_up_btn = st.button("🔥 Прогреть модель")
    
    st.markdown("---")
    st.caption("""
    **Легенда узлов:**  
    🔵 США (ядро)  
    🔴 Китай (ядро)  
    🟢 РФ (периферия, изгнание → ⚪)  
    🟠 Другие Core-агенты  
    🟡 Large  
    ⚪ Периферия
    """)

# Инициализация модели для визуализации
if 'viz_sim' not in st.session_state:
    st.session_state.viz_sim = HybridGFNSBS(n_agents=n_agents, seed=seed, disable_bailout=False)
    st.session_state.viz_sim.calibrate_to_2026_reality()
    st.session_state.viz_sim.warm_up(30)

# Кнопка прогрева
if warm_up_btn:
    with st.spinner("Прогрев модели..."):
        st.session_state.viz_sim = HybridGFNSBS(n_agents=n_agents, seed=seed, disable_bailout=False)
        st.session_state.viz_sim.calibrate_to_2026_reality()
        st.session_state.viz_sim.warm_up(50)
    st.success("Модель прогрета!")
    st.rerun()

# ====================== ВИЗУАЛИЗАЦИЯ СЕТИ ======================
st.divider()
st.header("🕸️ Сетевая структура GFN")

col_net1, col_net2 = st.columns([3, 2])

with col_net1:
    layer_choice = st.selectbox("Выберите слой", list(st.session_state.viz_sim.layers.keys()), 
                                 format_func=lambda x: x.capitalize())
    G_selected = st.session_state.viz_sim.layers[layer_choice]
    
    fig_network = create_network_plot(G_selected, st.session_state.viz_sim, layer_choice)
    st.plotly_chart(fig_network, use_container_width=True)

with col_net2:
    st.subheader("📊 Позиции в GFN")
    
    geo_ids = [st.session_state.viz_sim.USA_ID, 
               st.session_state.viz_sim.CHINA_ID, 
               st.session_state.viz_sim.RUSSIA_ID]
    position_data = st.session_state.viz_sim.get_position_in_gfn(geo_ids)
    
    df_positions = pd.DataFrame([
        {
            "🌍 Агент": d['name'],
            "📌 Тип": d['type'].capitalize(),
            "💪 Сила": f"{d['strength']:.3f}",
            "🎯 Центральность": f"{d['degree_c']:.3f}",
            "🔗 Betweenness": f"{d['betweenness']:.4f}",
            "🕸️ Кластеризация": f"{d['clustering']:.3f}",
            "🌑 SBS доля": f"{d['sbs_share']:.2f}",
            "⚠️ Haircut": f"{d['haircut']:.3f}",
            "🚪 Изгнан": "⚠️ Да" if d['is_exiled'] else "Нет",
            "✅ Полноценное место": "✔️ Да" if d['full_member'] else "❌ Нет"
        } for d in position_data.values()
    ], index=[d['name'] for d in position_data.values()])
    
    st.dataframe(df_positions, use_container_width=True)
    
    st.info("""
    **🔍 Политэкономическая интерпретация:**
    
    - **США и Китай** обладают полноценным местом в GFN
    - **РФ** — периферийный актор с SBS долей 55%, повышенным haircut
    - **Борьба за место в GFN** объясняет внешнеполитическую стратегию РФ
    """)

# ====================== TWIN-SIMULATION ======================
if run_twin_btn:
    st.divider()
    st.header("📈 Twin-Simulation: Сравнение режимов 2026-2030")
    st.warning("""
    **⚠️ Реалистичная калибровка на 2026 год:**
    
    Начальное состояние отражает текущие реалии:
    - Долговой кризис, фрагментация, высокие ставки
    - SBS доля РФ выросла до 55% из-за изоляции
    - **Начальный CCI: 38** (зона повышенного риска)
    """)
    
    with st.spinner(f"Запуск {n_simulations} симуляций..."):
        results = run_twin_simulation_realistic(n_agents, forecast_years, seed, n_simulations)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['years'],
        y=results['with']['mean'],
        mode='lines+markers',
        name='Режим A: С вмешательством',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, symbol='circle'),
        error_y=dict(type='data', array=results['with']['std'], visible=True)
    ))
    
    fig.add_trace(go.Scatter(
        x=results['years'],
        y=results['without']['mean'],
        mode='lines+markers',
        name='Режим B: Без вмешательства',
        line=dict(color='#d62728', width=3, dash='dash'),
        marker=dict(size=8, symbol='square'),
        error_y=dict(type='data', array=results['without']['std'], visible=True)
    ))
    
    fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="darkred", opacity=0.2, 
                  annotation_text="Коллапс", annotation_position="top left")
    fig.add_hrect(y0=50, y1=70, line_width=0, fillcolor="red", opacity=0.15, 
                  annotation_text="Критическая нестабильность", annotation_position="top left")
    fig.add_hrect(y0=30, y1=50, line_width=0, fillcolor="orange", opacity=0.1, 
                  annotation_text="Повышенный риск (2026)", annotation_position="top left")
    fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.05, 
                  annotation_text="Стабильность", annotation_position="top left")
    
    fig.add_annotation(
        x=2026,
        y=results['initial_cci'],
        text=f"Начало 2026: CCI = {results['initial_cci']:.0f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-30,
        bgcolor="white",
        borderwidth=1
    )
    
    fig.update_layout(
        title="Twin-Simulation: Реалистичная калибровка на 2026 год",
        xaxis_title="Год",
        yaxis_title="CCI (0 — стабильно, 100 — коллапс)",
        yaxis=dict(range=[0, 100]),
        height=550,
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    final_with = results['with']['mean'][-1]
    final_without = results['without']['mean'][-1]
    
    col_int1, col_int2, col_int3 = st.columns(3)
    
    with col_int1:
        st.metric("Начальный CCI (2026)", f"{results['initial_cci']:.0f}")
    with col_int2:
        st.metric("CCI 2030 (Режим A)", f"{final_with:.0f}")
    with col_int3:
        st.metric("CCI 2030 (Режим B)", f"{final_without:.0f}")
    
    st.info(f"""
    **Ключевые выводы:**
    
    1. **Начальное состояние (2026):** CCI = {results['initial_cci']:.0f} — повышенный риск.
       Отражает долговой кризис, фрагментацию и SBS-уязвимость.
    
    2. **Траектория без вмешательства:** CCI растёт до {final_without:.0f} к 2030 году.
    
    3. **Эффект интервенции:** CCI снижается на {final_without - final_with:.0f} пунктов.
    
    4. **Системный вывод:** Даже при спасении CCI остаётся >30 — структурная уязвимость.
    """)

# ====================== ТЕКУЩЕЕ СОСТОЯНИЕ ======================
st.divider()
st.header("📊 Текущее состояние модели")

col_state1, col_state2 = st.columns(2)

with col_state1:
    metrics_data = {
        "Метрика": ["Текущий CCI", "Ponzi (ср.)", "Haircut (ср.)", "Доверие", "SBS крах", "Bailout", "Россия изгнана"],
        "Значение": [
            f"{st.session_state.viz_sim.cci_history[-1]:.1f}" if st.session_state.viz_sim.cci_history else "N/A",
            f"{np.mean([a.ponzi_level for a in st.session_state.viz_sim.agents.values() if not a.is_exiled]):.3f}",
            f"{np.mean([a.haircut for a in st.session_state.viz_sim.agents.values() if a.sbs_share > 0.2]):.3f}" if any(a.sbs_share > 0.2 for a in st.session_state.viz_sim.agents.values()) else "N/A",
            f"{st.session_state.viz_sim.systemic_confidence:.1f}",
            "Да" if st.session_state.viz_sim.sbs_crash_triggered else "Нет",
            "Да" if st.session_state.viz_sim.bailout_active else "Нет",
            "⚠️ Да" if st.session_state.viz_sim.russia_exiled else "Нет"
        ]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

with col_state2:
    if len(st.session_state.viz_sim.cci_history) > 20:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            y=st.session_state.viz_sim.cci_history[-200:],
            mode='lines',
            name='CCI',
            line=dict(color='#4B0082', width=2)
        ))
        fig_hist.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Коллапс")
        fig_hist.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Критично")
        fig_hist.add_hline(y=30, line_dash="dash", line_color="yellow", annotation_text="Повышенный риск")
        fig_hist.update_layout(title="Динамика CCI", height=250, xaxis_title="Шаг")
        st.plotly_chart(fig_hist, use_container_width=True)

st.caption("""
🏷️ **Модель:** Гибридная (ABM + SDM) с калибровкой на 2026 год.
🎯 **Ключевой результат:** Количественная оценка дивергенции между режимами.
🔬 **Россия:** SBS доля 55%, периферийный агент, уязвим к изгнанию.
""")