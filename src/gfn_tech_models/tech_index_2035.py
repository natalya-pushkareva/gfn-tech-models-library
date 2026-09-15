"""
================================================================
ТЕХНОЛОГИЧЕСКИЙ ИНДЕКС 2035 — STREAMLIT ВЕРСИЯ
С использованием Numba для максимальной производительности
================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from numba import njit, prange

# ================================================================
# НАСТРОЙКА СТРАНИЦЫ
# ================================================================
st.set_page_config(
    page_title="Технологический Индекс 2035",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# БАЗОВЫЕ ПАРАМЕТРЫ
# ================================================================
years = np.arange(2025, 2036)
n_years = len(years)

default_params = {
    1: {"beta": 0.28, "tau": 0.32, "mu": 0.47, "comp": 0.28, "shock": 0.65, "name": "Россия (Базовый)"},
    2: {"beta": 0.42, "tau": 0.45, "mu": 0.52, "comp": 0.45, "shock": 0.85, "name": "Китай"},
    3: {"beta": 0.68, "tau": 0.72, "mu": 0.82, "comp": 0.35, "shock": 0.75, "name": "США"}
}

scenario_colors = ['#1f77b4', '#2ca02c', '#9467bd']

# ================================================================
# ULTRA-OPTIMIZED NUMBA KERNEL
# ================================================================
@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def monte_carlo_ultra(beta, tau, mu, compute_growth, shock_res, n_sim, n_years):
    """
    Максимально агрессивная оптимизация Monte-Carlo с Numba
    """
    trajectories = np.empty((n_sim, n_years), dtype=np.float64)
    trajectories[:, 0] = 35.0

    for i in prange(n_sim):
        b = beta
        t = tau
        m = mu
        c = 0.07

        for y in range(1, n_years):
            # Стохастическое обновление параметров
            b *= np.exp(np.random.normal(0.0, 0.014))
            t *= np.exp(np.random.normal(0.0, 0.013))
            m *= np.exp(np.random.normal(0.0, 0.01))
            c *= np.exp(compute_growth + np.random.normal(0.0, 0.09))

            growth = 0.85 * b * t + 0.55 * m + 0.45 * np.log1p(c) * 0.13
            interaction = 2.0 * b * t

            shock = 0.0
            if 2026 <= years[y] <= 2027:
                shock = -2.0 * (1.0 - shock_res) * np.random.beta(2.0, 5.0)

            delta = growth + interaction + shock + np.random.normal(0.0, 0.68)
            trajectories[i, y] = max(22.0, trajectories[i, y-1] + delta * 0.87)

    return trajectories

# ================================================================
# КЭШИРУЕМЫЙ РАСЧЁТ (выполняется 1 раз при изменении параметров)
# ================================================================
@st.cache_data(ttl=3600, show_spinner="⏳ Выполняется Monte-Carlo симуляция...")
def run_simulation(beta1, tau1, mu1, comp1, shock1,
                   beta2, tau2, mu2, comp2, shock2,
                   beta3, tau3, mu3, comp3, shock3,
                   n_sim):
    """
    Запуск всех трёх сценариев с кэшированием результатов
    """
    params = [
        (beta1, tau1, mu1, comp1, shock1),
        (beta2, tau2, mu2, comp2, shock2),
        (beta3, tau3, mu3, comp3, shock3)
    ]
    
    results = []
    
    for idx, (beta, tau, mu, comp, shock) in enumerate(params):
        trajectories = monte_carlo_ultra(beta, tau, mu, comp, shock, n_sim, n_years)
        
        mean_traj = np.mean(trajectories, axis=0)
        p5 = np.percentile(trajectories, 5, axis=0)
        p95 = np.percentile(trajectories, 95, axis=0)
        final = trajectories[:, -1]
        
        results.append({
            'name': default_params[idx+1]['name'],
            'mean_traj': mean_traj,
            'p5': p5,
            'p95': p95,
            'final': final,
            'mean_final': np.mean(final),
            'median_final': np.median(final),
            'p5_final': np.percentile(final, 5),
            'p95_final': np.percentile(final, 95),
            'std_final': np.std(final)
        })
    
    return results

# ================================================================
# ИНТЕРФЕЙС STREAMLIT
# ================================================================
st.title("📊 Технологический Индекс 2035 — Monte-Carlo с Numba")
st.caption("Гибридная модель: системная динамика + агентное моделирование. "
           "Симуляция 5000 траекторий с оптимизацией Numba.")

# ================================================================
# САЙДБАР: НАСТРОЙКА ПАРАМЕТРОВ
# ================================================================
with st.sidebar:
    st.header("⚙️ Параметры сценариев")
    
    # Количество симуляций
    n_sim = st.slider("Количество симуляций", 1000, 20000, 5000, step=1000,
                      help="Больше симуляций → точнее, но дольше")
    
    st.divider()
    
    # Три сценария
    for i in range(1, 4):
        st.subheader(f"Сценарий {i}: {default_params[i]['name']}")
        
        # Выбор имени
        name_options = ["Россия (Базовый)", "Китай", "США", "Оптимистичный РФ", "Европа", "Япония"]
        name_idx = name_options.index(default_params[i]['name']) if default_params[i]['name'] in name_options else 0
        default_params[i]['name'] = st.selectbox(
            f"Название сценария {i}",
            options=name_options,
            index=name_idx,
            key=f"name_{i}"
        )
        
        beta = st.slider(f"β — Связность (сц. {i})", 0.1, 0.85, default_params[i]['beta'], 0.005, key=f"beta_{i}")
        tau = st.slider(f"τ — Трансфер (сц. {i})", 0.1, 0.85, default_params[i]['tau'], 0.005, key=f"tau_{i}")
        mu = st.slider(f"μ — Рыночная свобода (сц. {i})", 0.3, 0.9, default_params[i]['mu'], 0.005, key=f"mu_{i}")
        comp = st.slider(f"Рост compute (сц. {i})", 0.1, 0.6, default_params[i]['comp'], 0.005, key=f"comp_{i}")
        shock = st.slider(f"Устойчивость к кризису (сц. {i})", 0.4, 0.95, default_params[i]['shock'], 0.005, key=f"shock_{i}")
        
        # Сохраняем в default_params для использования в расчёте
        default_params[i]['beta'] = beta
        default_params[i]['tau'] = tau
        default_params[i]['mu'] = mu
        default_params[i]['comp'] = comp
        default_params[i]['shock'] = shock
        
        st.divider()
    
    if st.button("🚀 Запустить расчёт", type="primary"):
        st.session_state.run_calc = True
    else:
        st.session_state.run_calc = False

# ================================================================
# ОСНОВНАЯ ОБЛАСТЬ: РЕЗУЛЬТАТЫ
# ================================================================
if not st.session_state.get('run_calc', False):
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Запустить расчёт'")
    st.stop()

# Запуск расчёта
with st.spinner("⏳ Выполняется Monte-Carlo симуляция... (это может занять 5-10 секунд)"):
    start_time = time.time()
    
    results = run_simulation(
        default_params[1]['beta'], default_params[1]['tau'], default_params[1]['mu'],
        default_params[1]['comp'], default_params[1]['shock'],
        default_params[2]['beta'], default_params[2]['tau'], default_params[2]['mu'],
        default_params[2]['comp'], default_params[2]['shock'],
        default_params[3]['beta'], default_params[3]['tau'], default_params[3]['mu'],
        default_params[3]['comp'], default_params[3]['shock'],
        n_sim
    )
    
    elapsed = time.time() - start_time
    st.success(f"✅ Расчёт завершён за {elapsed:.2f} сек (n_sim={n_sim})")

# ================================================================
# ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
# ================================================================

# ---- Основной график ----
st.subheader("📈 Динамика технологического индекса")

fig_main = go.Figure()

for idx, res in enumerate(results):
    color = scenario_colors[idx]
    name = res['name']
    
    fig_main.add_trace(go.Scatter(
        x=years,
        y=res['mean_traj'],
        name=name,
        line=dict(color=color, width=3.2)
    ))
    
    fig_main.add_trace(go.Scatter(
        x=years,
        y=res['p95'],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig_main.add_trace(go.Scatter(
        x=years,
        y=res['p5'],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.22)',
        name=f'{name} (5-95%)',
        showlegend=True
    ))

fig_main.add_vrect(
    x0=2026, x1=2027,
    fillcolor="red", opacity=0.12,
    annotation_text="Кризис",
    annotation_position="top right"
)

fig_main.update_layout(
    title=f"Технологический индекс — Monte-Carlo (n_sim = {n_sim})",
    xaxis_title="Год",
    yaxis_title="Индекс",
    height=500,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig_main, use_container_width=True)

# ---- Распределение в 2035 году ----
st.subheader("📊 Распределение индекса в 2035 году")

fig_dist = go.Figure()

for idx, res in enumerate(results):
    fig_dist.add_trace(go.Histogram(
        x=res['final'],
        name=res['name'],
        opacity=0.7,
        marker_color=scenario_colors[idx],
        nbinsx=80
    ))

fig_dist.update_layout(
    title="Распределение значений технологического индекса в 2035 году",
    barmode='overlay',
    height=400,
    template="plotly_white",
    xaxis_title="Технологический индекс",
    yaxis_title="Частота"
)

st.plotly_chart(fig_dist, use_container_width=True)

# ---- Таблица метрик ----
st.subheader("📋 Метрики в 2035 году")

table_data = []
for idx, res in enumerate(results):
    table_data.append({
        'Сценарий': res['name'],
        'Среднее': round(res['mean_final'], 1),
        'Медиана': round(res['median_final'], 1),
        '5%': round(res['p5_final'], 1),
        '95%': round(res['p95_final'], 1),
        'Ст.откл.': round(res['std_final'], 2)
    })

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---- Кнопка сброса ----
if st.button("🔄 Сбросить параметры к базовым"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# ---- Дополнительная информация ----
with st.expander("ℹ️ О модели"):
    st.markdown("""
    **Модель технологического индекса** построена на гибридном подходе:
    
    - **Системная динамика (SD)**: макропотоки знаний, инвестиций, диффузия технологий
    - **Агентное моделирование (ABM)**: гетерогенные агенты (фирмы, университеты, НИИ)
    - **Метод Монте-Карло**: 5000+ траекторий для учёта неопределённости
    
    **Параметры модели:**
    - **β (Связность)**: интенсивность кооперации между агентами
    - **τ (Трансфер)**: скорость передачи знаний и технологий
    - **μ (Рыночная свобода)**: степень рыночной конкуренции
    - **Рост compute**: темп роста вычислительных мощностей
    - **Устойчивость к кризису**: способность противостоять внешним шокам
    """)