"""
RUS-GFN-TSI-2035 CASCADE — Исправленная версия
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.stats import beta, poisson, norm
from collections import deque, defaultdict
import random
import networkx as nx
from itertools import combinations
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RUS-GFN-TSI-2035 CASCADE", layout="wide")

# ====================== 1. КАСКАДНАЯ МОДЕЛЬ ГАЙ-КАПАДИА ======================

class CascadingNetwork:
    """
    Реализация каскадной модели распространения дефолтов
    Основана на работе Gai & Kapadia (2010)
    """
    
    def __init__(self, n_nodes=90, seed=42):
        self.n_nodes = n_nodes
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Пороги уязвимости (thresholds) - генерируются из бета-распределения
        self.thresholds = beta.rvs(2, 5, size=n_nodes) * 0.5 + 0.1  # [0.1, 0.6]
        
        # Начальное состояние: 0 - здоров, 1 - дефолтен
        self.defaulted = np.zeros(n_nodes, dtype=int)
        
        # Матрица взаимных обязательств (стохастическая)
        self.exposures = None
        self._initialize_network()
        
        # История каскадов
        self.cascade_history = []
        
    def _initialize_network(self):
        """Инициализация стохастического графа с LTIA-структурой"""
        degrees = poisson.rvs(mu=4, size=self.n_nodes)
        degrees = np.clip(degrees, 1, 15)
        
        self.G = nx.configuration_model(degrees, seed=self.seed)
        self.G = nx.Graph(self.G)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.G = nx.Graph(self.G)
        
        for u, v in self.G.edges():
            weight = beta.rvs(2, 3) * 0.3 + 0.1
            self.G[u][v]['weight'] = weight
            self.G[u][v]['exposure'] = weight * random.uniform(0.5, 1.5)
        
        self.exposures = np.zeros((self.n_nodes, self.n_nodes))
        for u, v in self.G.edges():
            w = self.G[u][v]['exposure']
            self.exposures[u][v] = w
            self.exposures[v][u] = w
        
        self.capital = np.ones(self.n_nodes) * 100
        self.capital *= (1 + norm.rvs(0, 0.2, size=self.n_nodes))
        
        self.russia_idx = 2
        self.thresholds[self.russia_idx] = 0.25
        self.capital[self.russia_idx] = 80
        
        self.thresholds[0] = 0.45
        self.thresholds[1] = 0.40
        self.capital[0] = 150
        self.capital[1] = 130
        
    def propagate_cascade(self, initial_shock_nodes=None, shock_strength=0.1):
        """Запуск каскадного распространения дефолтов"""
        self.defaulted = np.zeros(self.n_nodes, dtype=int)
        self.cascade_history = []
        
        if initial_shock_nodes is None:
            initial_shock_nodes = np.random.choice(self.n_nodes, size=3, replace=False)
        
        for node in initial_shock_nodes:
            loss = self.capital[node] * shock_strength
            if loss > self.capital[node] * self.thresholds[node]:
                self.defaulted[node] = 1
        
        self.cascade_history.append(self.defaulted.copy())
        
        max_iterations = 50
        for iteration in range(max_iterations):
            new_defaults = np.zeros(self.n_nodes, dtype=int)
            
            for i in range(self.n_nodes):
                if self.defaulted[i] == 0:
                    losses = 0
                    for j in range(self.n_nodes):
                        if self.defaulted[j] == 1 and self.exposures[i][j] > 0:
                            losses += self.exposures[i][j] * self.capital[j] * 0.5
                    
                    if losses > self.capital[i] * self.thresholds[i]:
                        new_defaults[i] = 1
            
            self.defaulted = np.maximum(self.defaulted, new_defaults)
            self.cascade_history.append(self.defaulted.copy())
            
            if np.sum(new_defaults) == 0:
                break
        
        return self.defaulted, self.cascade_history
    
    def get_cascade_metrics(self):
        """Получение метрик каскада"""
        if not self.cascade_history:
            return {
                'total_defaults': 0,
                'cascade_length': 0,
                'max_defaults_rate': 0,
                'systemic_risk': 0,
                'russia_impact': 0,
                'final_state': np.zeros(self.n_nodes)
            }
        
        final_state = self.cascade_history[-1]
        total_defaults = np.sum(final_state)
        cascade_length = len(self.cascade_history)
        max_defaults = max([np.sum(state) for state in self.cascade_history])
        systemic_risk = total_defaults / self.n_nodes
        russia_impact = final_state[self.russia_idx] if self.russia_idx < len(final_state) else 0
        
        return {
            'total_defaults': total_defaults,
            'cascade_length': cascade_length,
            'max_defaults_rate': max_defaults / self.n_nodes,
            'systemic_risk': systemic_risk,
            'russia_impact': russia_impact,
            'final_state': final_state
        }
    
    def get_russia_exposure(self):
        """Оценка влияния каскада на Россию через третьи страны"""
        russia_idx = self.russia_idx
        russia_neighbors = list(self.G.neighbors(russia_idx))
        russia_exposure = sum([self.exposures[russia_idx][n] for n in russia_neighbors])
        
        indirect_risk = 0
        for neighbor in russia_neighbors:
            neighbors_of_neighbor = list(self.G.neighbors(neighbor))
            neighbors_of_neighbor = [n for n in neighbors_of_neighbor if n != russia_idx]
            
            for n2 in neighbors_of_neighbor:
                indirect_risk += (self.exposures[russia_idx][neighbor] * 
                                 self.exposures[neighbor][n2] * 0.1)
        
        return {
            'direct_neighbors': len(russia_neighbors),
            'direct_exposure': russia_exposure,
            'indirect_risk': indirect_risk,
            'total_risk': russia_exposure + indirect_risk,
            'vulnerability_score': min(1.0, (russia_exposure + indirect_risk * 0.5) / 10)
        }

# ====================== 2. РАСШИРЕННАЯ GFN С КАСКАДНЫМ МЕХАНИЗМОМ ======================

class AdvancedGFNWithCascade:
    def __init__(self, n_agents=90, seed=42):
        self.n_agents = n_agents
        self.seed = seed
        self.cascade_network = CascadingNetwork(n_agents, seed)
        
        self.agents = {}
        self.create_agents()
        self.assign_geo_attributes()
        
        self.history = []
        self.russia_history = []
        self.cascade_metrics_history = []
        
    def create_agents(self):
        for i in range(self.n_agents):
            if i < 7:
                typ, s = 'core', random.uniform(0.84, 1.0)
            elif i < 22:
                typ, s = 'large', random.uniform(0.52, 0.84)
            else:
                typ, s = 'peripheral', random.uniform(0.08, 0.48)
            self.agents[i] = {
                'type': typ,
                'strength': s,
                'risk_appetite': 0.35,
                'ponzi_level': 0.0,
                'contagion_exposure': 0.0,
                'defaulted': False,
                'geo': None
            }
    
    def assign_geo_attributes(self):
        self.agents[0]['type'] = 'core'
        self.agents[0]['strength'] = 0.97
        self.agents[0]['geo'] = 'USA'
        
        self.agents[1]['type'] = 'core'
        self.agents[1]['strength'] = 0.93
        self.agents[1]['geo'] = 'China'
        
        self.agents[2]['type'] = 'peripheral'
        self.agents[2]['strength'] = 0.31
        self.agents[2]['geo'] = 'Russia'
        self.agents[2]['sanctions_impact'] = 0.0
        self.agents[2]['isolation'] = 0.0
    
    def step(self, external_shock_strength=0.1, russia_sanctions=0.0):
        shock_nodes = np.random.choice(self.n_agents, size=max(1, int(self.n_agents * external_shock_strength)), replace=False)
        defaults, history = self.cascade_network.propagate_cascade(
            initial_shock_nodes=shock_nodes,
            shock_strength=external_shock_strength
        )
        
        cascade_metrics = self.cascade_network.get_cascade_metrics()
        
        for i in range(self.n_agents):
            if defaults[i] == 1:
                self.agents[i]['defaulted'] = True
                self.agents[i]['strength'] *= 0.7
                self.agents[i]['risk_appetite'] += 0.1
            else:
                self.agents[i]['strength'] = min(1.0, self.agents[i]['strength'] * 1.01)
                self.agents[i]['risk_appetite'] = max(0.2, self.agents[i]['risk_appetite'] - 0.01)
            
            self.agents[i]['ponzi_level'] = max(0.0, self.agents[i]['risk_appetite'] - 0.57)
        
        russia = self.agents[2]
        russia['sanctions_impact'] = min(0.8, russia.get('sanctions_impact', 0) + 
                                        russia_sanctions * 0.05 + external_shock_strength * 0.03)
        russia['isolation'] = min(0.9, russia.get('isolation', 0) + 
                                cascade_metrics['systemic_risk'] * 0.02 + russia_sanctions * 0.04)
        
        if russia.get('sanctions_impact', 0) > 0.3:
            russia['strength'] *= (1 - 0.005 * russia['sanctions_impact'])
        
        self.record_state(cascade_metrics)
        return cascade_metrics
    
    def record_state(self, cascade_metrics):
        avg_risk = np.mean([a['risk_appetite'] for a in self.agents.values()])
        defaulted_count = sum([1 for a in self.agents.values() if a['defaulted']])
        
        state = {
            'avg_risk': avg_risk,
            'defaulted_count': defaulted_count,
            'default_rate': defaulted_count / self.n_agents,
            'cascade_length': cascade_metrics['cascade_length'],
            'systemic_risk': cascade_metrics['systemic_risk'],
            'russia_strength': self.agents[2]['strength'],
            'russia_sanctions': self.agents[2].get('sanctions_impact', 0),
            'russia_isolation': self.agents[2].get('isolation', 0)
        }
        self.history.append(state)
        
        russia_state = {
            'strength': self.agents[2]['strength'],
            'risk_appetite': self.agents[2]['risk_appetite'],
            'ponzi_level': self.agents[2]['ponzi_level'],
            'sanctions_impact': self.agents[2].get('sanctions_impact', 0),
            'isolation': self.agents[2].get('isolation', 0),
            'defaulted': self.agents[2]['defaulted']
        }
        self.russia_history.append(russia_state)
        self.cascade_metrics_history.append(cascade_metrics)
    
    def get_gfn_metrics(self):
        if not self.history:
            return self._get_default_metrics()
        
        last = self.history[-1]
        cascade_last = self.cascade_metrics_history[-1] if self.cascade_metrics_history else {}
        russia_last = self.russia_history[-1] if self.russia_history else {}
        
        contagion_potential = min(1.0, 
            last['avg_risk'] * 0.3 + 
            last['systemic_risk'] * 0.3 + 
            russia_last.get('isolation', 0) * 0.2 +
            last['default_rate'] * 0.2
        )
        
        return {
            'gfn_risk': last['avg_risk'],
            'systemic_pressure': last['systemic_risk'],
            'layer_instability': last['avg_risk'] * 0.8,
            'contagion_potential': contagion_potential,
            'default_rate': last['default_rate'],
            'cascade_length': last['cascade_length'],
            'russia_sanctions': russia_last.get('sanctions_impact', 0),
            'russia_isolation': russia_last.get('isolation', 0),
            'russia_strength': last['russia_strength'],
            'russia_defaulted': russia_last.get('defaulted', False)
        }
    
    def _get_default_metrics(self):
        return {
            'gfn_risk': 0.55,
            'systemic_pressure': 0.3,
            'layer_instability': 0.4,
            'contagion_potential': 0.2,
            'default_rate': 0.0,
            'cascade_length': 0,
            'russia_sanctions': 0.0,
            'russia_isolation': 0.0,
            'russia_strength': 0.31,
            'russia_defaulted': False
        }

# ====================== 3. SDM С КАСКАДНЫМИ ЭФФЕКТАМИ ======================

class RussiaSDM_WithCascade:
    def __init__(self, delay_years=3.5):
        self.delay_years = delay_years
        self.history = None
        self.gfn_impact_history = []
        self.cascade_shock_history = []
    
    def calculate_gfn_impact(self, gfn_metrics):
        if gfn_metrics is None:
            return 0.0
        
        base_impact = -(
            gfn_metrics.get('gfn_risk', 0.55) * 8.0 +
            gfn_metrics.get('systemic_pressure', 0.3) * 6.0 +
            gfn_metrics.get('contagion_potential', 0.2) * 5.0 +
            gfn_metrics.get('russia_sanctions', 0) * 12.0 +
            gfn_metrics.get('russia_isolation', 0) * 7.0
        )
        
        cascade_impact = -(
            gfn_metrics.get('default_rate', 0) * 15.0 +
            gfn_metrics.get('cascade_length', 0) * 0.5
        )
        
        return base_impact + cascade_impact * 0.6
    
    def _derivative(self, t, y, params, gfn_metrics=None, calib=None):
        if calib is None:
            calib = {'tsi_scale': 0.94, 'tech_scale': 0.89, 'trust_scale': 0.96}
        
        K, TSI, Inf, Prod, Tech, Gini, Trust = y
        kr, mil, sanc, inv_p, t_acc, a_corr = params
        
        delay_steps = int(self.delay_years / 0.25)
        past_TSI = self.history[-delay_steps][1] if len(self.history) > delay_steps else TSI
        past_Trust = self.history[-delay_steps][6] if len(self.history) > delay_steps else Trust
        
        gfn_impact = self.calculate_gfn_impact(gfn_metrics) if gfn_metrics else 0
        
        cascade_tech_penalty = gfn_metrics.get('default_rate', 0) * 0.3 if gfn_metrics else 0
        tech_global_factor = 1.0 - (gfn_metrics.get('russia_isolation', 0) * 0.5 + cascade_tech_penalty)
        
        dTech = (calib['tech_scale'] * 0.0135 * t_acc * Tech * (1 - Tech / 2.35) * 
                (0.76 + 0.24 * past_Trust) * tech_global_factor)
        
        tsi_base = 51.5 + 17.5 * (1 - sanc**1.65) * (1 - a_corr**0.88) + 10.5 * t_acc
        tsi_gfn_adjustment = gfn_impact * 0.3
        tsi_cascade_adjustment = -gfn_metrics.get('default_rate', 0) * 8.0 if gfn_metrics else 0
        tsi_target = calib['tsi_scale'] * (tsi_base + tsi_gfn_adjustment + tsi_cascade_adjustment)
        
        dTSI = (0.108 * (tsi_target - TSI) + 0.215 * (Tech - 1.0) + 
               0.07 * (past_TSI - TSI) + gfn_impact * 0.05)
        
        cascade_inflation_shock = gfn_metrics.get('default_rate', 0) * 1.0 if gfn_metrics else 0
        gfn_inflation_shock = gfn_metrics.get('gfn_risk', 0) * 0.5 if gfn_metrics else 0
        dInf = 0.079 * (4.3 + 4.0 * sanc + 0.68 * mil + gfn_inflation_shock + cascade_inflation_shock - Inf)
        
        dProd = (Prod * 0.0098 * (1 - 0.34 * sanc) * (1 + 0.11 * (TSI - 52)/50) - 
                Prod * 0.0075 - (gfn_metrics.get('contagion_potential', 0) * 0.003 if gfn_metrics else 0) -
                gfn_metrics.get('default_rate', 0) * 0.005 if gfn_metrics else 0)
        
        dGini = 0.037 * (0.336 + 0.058 * mil + 0.055 * sanc - 0.026 * t_acc - Gini)
        
        trust_global_factor = 1.0 - gfn_metrics.get('systemic_pressure', 0) * 0.3 if gfn_metrics else 0
        trust_cascade_factor = 1.0 - gfn_metrics.get('default_rate', 0) * 0.2 if gfn_metrics else 0
        dTrust = (calib['trust_scale'] * 0.053 * 
                 (0.485 - 0.39 * sanc - 0.28 * a_corr + 0.135 * t_acc - Trust) * 
                 trust_global_factor * trust_cascade_factor +
                 0.095 * (past_Trust - Trust))
        
        real_rate = (kr - Inf) / 100
        inv_rate = np.clip(0.168 - 0.315 * real_rate - 0.019 * mil + 0.011 * (inv_p/100) + 
                          0.0045 * (TSI-52)/52, 0.08, 0.25)
        dK = K * inv_rate - K * 0.055
        
        if gfn_metrics:
            self.cascade_shock_history.append(gfn_metrics.get('default_rate', 0))
        
        dy = [dK, dTSI, dInf, dProd, dTech, dGini, dTrust]
        self.history.append(y.copy())
        return dy
    
    def run(self, params, gfn_metrics_series, years=10, calib=None):
        initial = np.array([108.0, 54.0, 8.5, 1.0, 1.0, 0.33, 0.45], dtype=float)
        self.history = deque(maxlen=400)
        self.history.append(initial.copy())
        self.gfn_impact_history = []
        self.cascade_shock_history = []
        
        time_points = np.linspace(0, years, 21)
        
        def get_gfn_at_time(t):
            if not gfn_metrics_series:
                return None
            idx = int(t / years * (len(gfn_metrics_series) - 1))
            idx = min(idx, len(gfn_metrics_series) - 1)
            return gfn_metrics_series[idx]
        
        def func_wrapper(t, y):
            gfn_metrics = get_gfn_at_time(t)
            return self._derivative(t, y, params, gfn_metrics, calib)
        
        sol = solve_ivp(
            fun=func_wrapper,
            t_span=(0, years),
            y0=initial,
            method='RK45',
            t_eval=time_points,
            rtol=1e-3, atol=1e-5, max_step=0.5
        )
        
        for t in sol.t:
            gfn_metrics = get_gfn_at_time(t)
            if gfn_metrics:
                self.gfn_impact_history.append(self.calculate_gfn_impact(gfn_metrics))
            else:
                self.gfn_impact_history.append(0.0)
        
        # Убеждаемся, что длины массивов совпадают
        cascade_shock_array = np.array(self.cascade_shock_history)
        if len(cascade_shock_array) < len(sol.t):
            # Дополняем недостающие значения
            padding = np.zeros(len(sol.t) - len(cascade_shock_array))
            cascade_shock_array = np.concatenate([cascade_shock_array, padding])
        elif len(cascade_shock_array) > len(sol.t):
            # Обрезаем лишние
            cascade_shock_array = cascade_shock_array[:len(sol.t)]
        
        return {
            'time': sol.t + 2025,
            'K': sol.y[0],
            'TSI': sol.y[1],
            'Inf': sol.y[2],
            'Prod': sol.y[3],
            'Tech': sol.y[4],
            'Gini': sol.y[5],
            'Trust': sol.y[6],
            'gfn_impact': np.array(self.gfn_impact_history),
            'cascade_shock': cascade_shock_array
        }

# ====================== 4. ГЕНЕРАЦИЯ ДАННЫХ ======================

def generate_gfn_data_with_cascade(years=10, sanctions_intensity=0.6, shock_probability=0.3):
    gfn = AdvancedGFNWithCascade(90)
    
    for _ in range(5):
        gfn.step(external_shock_strength=0.05)
    
    gfn_series = []
    for year in range(years + 1):
        sanctions_year = sanctions_intensity * (1 + year * 0.12)
        sanctions_year = min(0.9, sanctions_year + np.random.normal(0, 0.03))
        
        shock_strength = 0.05 + year * 0.02 + np.random.uniform(0, 0.03)
        shock_strength = min(0.3, shock_strength)
        
        if random.random() < shock_probability:
            shock_strength = min(0.5, shock_strength + 0.1)
        
        cascade_metrics = gfn.step(
            external_shock_strength=shock_strength,
            russia_sanctions=sanctions_year
        )
        
        metrics = gfn.get_gfn_metrics()
        metrics.update({
            'shock_strength': shock_strength,
            'cascade_defaults': cascade_metrics['total_defaults'],
            'cascade_length': cascade_metrics['cascade_length']
        })
        gfn_series.append(metrics)
    
    return gfn_series, gfn

# ====================== 5. UI ======================

st.title("🇷🇺 RUS-GFN-TSI-2035 CASCADE")
st.markdown("**Гибридная модель с каскадным распространением дефолтов**")

with st.sidebar:
    st.header("🇷🇺 Параметры России")
    key_rate = st.slider("Ключевая ставка (%)", 12.0, 25.0, 16.5)
    mil_share = st.slider("Военные расходы (% ВВП)", 4.0, 10.0, 6.5)
    sanctions = st.slider("Внутренние санкции", 0.4, 1.0, 0.62)
    inv_push = st.slider("Инвестиции (% ВВП)", 1.0, 6.0, 2.3)
    tech_access = st.slider("Доступ к технологиям", 0.2, 0.9, 0.45)
    anti_corr = st.slider("Антикоррупция", 0.1, 0.8, 0.38)
    
    st.header("🌐 Каскадные параметры")
    global_sanctions = st.slider("Интенсивность санкций против РФ", 0.1, 1.0, 0.6)
    shock_probability = st.slider("Вероятность системного шока", 0.1, 0.7, 0.3)
    cascade_amplification = st.slider("Амплификация каскада", 1.0, 3.0, 1.5)
    
    run_button = st.button("🚀 Запустить каскадную модель", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Выполняется расчет с каскадным распространением..."):
        params = (key_rate, mil_share, sanctions, inv_push, tech_access, anti_corr)
        
        gfn_series, gfn = generate_gfn_data_with_cascade(
            years=10, 
            sanctions_intensity=global_sanctions,
            shock_probability=shock_probability
        )
        
        for i in range(len(gfn_series)):
            if i > 2:
                gfn_series[i]['default_rate'] = min(0.5, gfn_series[i]['default_rate'] * cascade_amplification)
                gfn_series[i]['contagion_potential'] = min(1.0, gfn_series[i]['contagion_potential'] * cascade_amplification * 0.8)
        
        sdm = RussiaSDM_WithCascade()
        result = sdm.run(params, gfn_series, years=10)
    
    # ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ====================
    
    st.subheader("📊 Ключевые показатели 2035 с каскадными эффектами")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    tsi_value = result['TSI'][-1]
    tsi_delta = result['TSI'][-1] - result['TSI'][0]
    col1.metric("TSI 2035", f"{tsi_value:.1f}", delta=f"{tsi_delta:.1f}")
    
    gdp_value = result['K'][-1] * 1.8
    gdp_delta = gdp_value - result['K'][0] * 1.8
    col2.metric("ВВП (трлн ₽)", f"{gdp_value:.0f}", delta=f"{gdp_delta:.0f}")
    
    inf_value = result['Inf'][-1]
    col3.metric("Инфляция (%)", f"{inf_value:.1f}")
    
    last_gfn = gfn_series[-1] if gfn_series else {}
    default_rate = last_gfn.get('default_rate', 0) * 100
    col4.metric("Уровень дефолтов", f"{default_rate:.1f}%")
    
    cascade_length = last_gfn.get('cascade_length', 0)
    col5.metric("Длина каскада", f"{cascade_length}")
    
    # Основная визуализация
    st.subheader("📈 Прогноз с каскадными эффектами")
    
    fig1 = make_subplots(rows=3, cols=3,
                         subplot_titles=(
                             "TSI (с каскадом)", "ВВП", "Инфляция",
                             "Технологии", "Производительность", "Доверие",
                             "Каскадный шок", "Уровень дефолтов", "Сравнение TSI"
                         ))
    
    # Row 1
    fig1.add_trace(go.Scatter(x=result['time'], y=result['TSI'], 
                              mode='lines+markers', name='TSI',
                              line=dict(color='green', width=3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=result['time'], y=result['K']*1.8,
                              mode='lines+markers', name='ВВП',
                              line=dict(color='blue')), row=1, col=2)
    fig1.add_trace(go.Scatter(x=result['time'], y=result['Inf'],
                              mode='lines+markers', name='Инфляция',
                              line=dict(color='red')), row=1, col=3)
    
    # Row 2
    fig1.add_trace(go.Scatter(x=result['time'], y=result['Tech'],
                              mode='lines+markers', name='Технологии',
                              line=dict(color='purple')), row=2, col=1)
    fig1.add_trace(go.Scatter(x=result['time'], y=result['Prod'],
                              mode='lines+markers', name='Производительность',
                              line=dict(color='orange')), row=2, col=2)
    fig1.add_trace(go.Scatter(x=result['time'], y=result['Trust'],
                              mode='lines+markers', name='Доверие',
                              line=dict(color='teal')), row=2, col=3)
    
    # Row 3 - Каскадные эффекты
    # Убеждаемся, что каскадный шок имеет правильную длину
    cascade_shock_display = result['cascade_shock'][1:] if len(result['cascade_shock']) > 1 else result['cascade_shock']
    time_display = result['time'][1:] if len(result['time']) > 1 else result['time']
    
    fig1.add_trace(go.Bar(x=time_display, y=cascade_shock_display,
                          name='Каскадный шок', marker_color='darkred', opacity=0.7),
                  row=3, col=1)
    
    default_rates = [m.get('default_rate', 0) * 100 for m in gfn_series]
    # Интерполируем для совпадения длины
    default_rates_interp = []
    for t in result['time']:
        idx = int((t - 2025) / 10 * (len(default_rates) - 1))
        idx = min(idx, len(default_rates) - 1)
        default_rates_interp.append(default_rates[idx])
    
    fig1.add_trace(go.Scatter(x=result['time'], y=default_rates_interp,
                              mode='lines+markers', name='Дефолты %',
                              line=dict(color='crimson', width=2)), row=3, col=2)
    
    # Сравнение TSI
    fig1.add_trace(go.Scatter(x=result['time'], y=result['TSI'],
                              mode='lines+markers', name='TSI (с каскадом)',
                              line=dict(color='green', width=3)), row=3, col=3)
    
    sdm_no_cascade = RussiaSDM_WithCascade()
    result_no_cascade = sdm_no_cascade.run(params, gfn_series, years=10)
    fig1.add_trace(go.Scatter(x=result_no_cascade['time'], y=result_no_cascade['TSI'],
                              mode='lines', name='TSI (без каскада)',
                              line=dict(color='gray', dash='dash')), row=3, col=3)
    
    fig1.update_layout(height=900, showlegend=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Каскадный анализ
    st.subheader("🌊 Анализ каскадного распространения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cascade_df = pd.DataFrame({
            'Год': result['time'][:len(gfn_series)],
            'Дефолты %': [m.get('default_rate', 0) * 100 for m in gfn_series],
            'Каскадный шок': result['cascade_shock'][:len(gfn_series)],
            'Потенциал заражения': [m.get('contagion_potential', 0) * 100 for m in gfn_series]
        })
        
        fig_cascade = go.Figure()
        fig_cascade.add_trace(go.Scatter(
            x=cascade_df['Год'], y=cascade_df['Дефолты %'],
            mode='lines+markers', name='Уровень дефолтов',
            line=dict(color='red', width=2)
        ))
        fig_cascade.add_trace(go.Scatter(
            x=cascade_df['Год'], y=cascade_df['Потенциал заражения'],
            mode='lines+markers', name='Потенциал заражения',
            line=dict(color='orange', width=2, dash='dash')
        ))
        fig_cascade.update_layout(title="Динамика каскадных эффектов", height=400)
        st.plotly_chart(fig_cascade, use_container_width=True)
    
    with col2:
        st.metric("Средний уровень дефолтов", f"{cascade_df['Дефолты %'].mean():.1f}%")
        st.metric("Максимальный уровень дефолтов", f"{cascade_df['Дефолты %'].max():.1f}%")
        st.metric("Средний потенциал заражения", f"{cascade_df['Потенциал заражения'].mean():.1f}%")
        
        total_impact = abs(result['gfn_impact'].sum())
        cascade_share = (abs(result['cascade_shock'].sum()) / total_impact * 100) if total_impact > 0 else 0
        st.metric("Доля каскадного эффекта", f"{min(100, cascade_share):.1f}%")
    
    # Детальный анализ
    st.subheader("📅 Детальный анализ по годам")
    
    gfn_risk_yearly = []
    default_rates_yearly = []
    for t in result['time']:
        idx = int((t - 2025) / 10 * (len(gfn_series) - 1))
        idx = min(idx, len(gfn_series) - 1)
        gfn_risk_yearly.append(gfn_series[idx].get('gfn_risk', 0.55) * 100)
        default_rates_yearly.append(gfn_series[idx].get('default_rate', 0) * 100)
    
    years_data = pd.DataFrame({
        'Год': result['time'].astype(int),
        'TSI': result['TSI'].round(1),
        'Влияние GFN': result['gfn_impact'].round(1),
        'Каскадный шок': result['cascade_shock'][:len(result['time'])].round(3),
        'GFN Risk %': np.array(gfn_risk_yearly).round(1),
        'Дефолты %': np.array(default_rates_yearly).round(1)
    })
    st.dataframe(years_data, use_container_width=True)
    
    st.download_button(
        label="📥 Скачать данные с каскадными эффектами (CSV)",
        data=years_data.to_csv(index=False).encode('utf-8'),
        file_name=f'rus_gfn_cascade_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
    
    # Визуализация сети
    st.subheader("🕸️ Визуализация каскадного распространения")
    
    if 'gfn' in locals():
        final_state = gfn.cascade_network.get_cascade_metrics()
        if 'final_state' in final_state:
            states = final_state['final_state']
            n_defaults = np.sum(states)
            st.metric("Количество дефолтов в сети", n_defaults)
            
            G = gfn.cascade_network.G
            pos = nx.spring_layout(G, seed=42)
            
            node_colors = ['red' if states[i] == 1 else 'blue' for i in range(len(states))]
            node_sizes = [30 if states[i] == 1 else 15 for i in range(len(states))]
            
            for i, size in enumerate(node_sizes):
                if i in [0, 1, 2]:
                    node_sizes[i] = 50 if states[i] == 1 else 35
            
            edge_x, edge_y = [], []
            for edge in G.edges():
                if edge[0] < len(pos) and edge[1] < len(pos):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            node_x, node_y, node_text = [], [], []
            for node in G.nodes():
                if node < len(pos):
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    geo_name = gfn.agents.get(node, {}).get('geo', '')
                    status = '🔴 ДЕФОЛТ' if states[node] == 1 else '🟢 ЗДОРОВ'
                    node_text.append(f"{geo_name} {node}: {status}")
            
            fig_network = go.Figure()
            
            fig_network.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.3, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
            
            fig_network.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=node_text,
                hoverinfo='text',
                showlegend=False
            ))
            
            fig_network.update_layout(
                title=f"Каскадное распространение (дефолтов: {n_defaults})",
                showlegend=False,
                hovermode='closest',
                height=500,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig_network, use_container_width=True)

st.caption("🇷🇺 RUS-GFN-TSI-2035 CASCADE | Модель с каскадным распространением дефолтов")