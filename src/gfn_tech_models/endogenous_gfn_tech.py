#!/usr/bin/env python3
"""
EndogenousGFN — Full Model with Technology Block + Tech Lag
===========================================================
Эндогенная эволюция Global Financial Network
+ технологическое соперничество (semiconductors, critical tech)
+ моделирование технологического отставания (tech lag)

Страны: USA, China, Russia, Eurozone, Japan, UK, India, Saudi

Возможности:
- tech_centrality, tech_dependence, tech_lag
- tech_sanctions как отдельный параметр
- Эндогенная динамика tech-переменных (growth, catch-up, sanction drag)
- Влияние технологий и lag на isolation, confidence и Position Score
- Сценарии: baseline, tech_escalation, tech_lag, dual_decoupling, russia_focus
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# ПАРАМЕТРЫ МОДЕЛИ
# ============================================================

@dataclass
class ModelParams:
    # ----- Links -----
    fit_strength_w: float = 0.45
    fit_confidence_w: float = 0.30
    fit_position_w: float = 0.25
    link_inertia: float = 0.75
    link_adjust: float = 0.25
    link_decay: float = 0.98
    sanctions_desired_mult: float = 0.50
    sanctions_current_mult: float = 0.12
    link_min: float = 0.02
    link_max: float = 0.95

    # ----- Isolation -----
    isol_normal_core: float = 0.55
    isol_normal_russia: float = 0.40
    isol_normal_periphery: float = 0.45
    isol_scale: float = 0.90
    isol_min: float = 0.03
    isol_max: float = 0.92

    # ----- Leverage / NPL / Confidence -----
    lev_base_growth: float = 0.009
    lev_isol_coef: float = 0.01
    lev_shock_coef: float = 0.06
    lev_min: float = 0.35
    lev_max: float = 0.90

    npl_adjust_speed: float = 0.25
    npl_min: float = 0.015
    npl_max: float = 0.25

    conf_shock_coef: float = 0.08
    conf_isol_coef: float = 0.05
    conf_npl_coef: float = 0.04
    conf_tech_coef: float = 0.06          # γ — влияние tech_sanctions на confidence
    conf_min: float = 0.15
    conf_max: float = 0.90

    # ----- Position Score -----
    isol_pen_coef: float = 0.35
    isol_pen_scale: float = 90.0
    lev_pen_coef: float = 0.14
    lev_pen_scale: float = 70.0
    conf_bonus_scale: float = 15.0
    residual_scale: float = 34.0
    residual_isol_mult: float = 0.5
    cascade_base: float = 7.5
    cascade_isol_mult: float = 1.2
    base_target: float = 48.0

    # Технологический эффект
    tech_centrality_bonus: float = 12.0   # α
    tech_dependence_penalty: float = 18.0 # β

    # Эндогенная динамика tech-переменных
    tech_c_growth: float = 0.012          # базовый рост centrality при высокой strength
    tech_c_sanction_drag: float = 0.025   # давление tech_sanctions на centrality
    tech_d_reduction: float = 0.018       # скорость снижения dependence (import substitution)
    tech_d_sanction_slow: float = 0.60    # насколько sanctions замедляют снижение dependence
    tech_c_min: float = 0.08
    tech_c_max: float = 0.96
    tech_d_min: float = 0.12
    tech_d_max: float = 0.90

    # Технологическое отставание (tech lag)
    # lag = max(0, frontier - own_centrality), frontier ≈ max(USA, Japan, Eurozone)
    tech_lag_penalty: float = 14.0        # штраф к Position Score от tech_lag
    tech_lag_conf_coef: float = 0.04      # влияние lag на confidence
    catchup_rate: float = 0.015           # скорость догоняющего развития (при низкой isolation)
    lag_isolation_mult: float = 0.8       # насколько isolation мешает catch-up
    frontier_countries: tuple = ("USA", "Japan", "Eurozone")

    # Якоря
    usa_anchor: float = 82.0
    usa_anchor_w: float = 0.40
    china_anchor: float = 55.0
    china_anchor_w: float = 0.35
    euro_anchor: float = 52.0
    euro_anchor_w: float = 0.30

    # Обновление Position
    pos_inertia: float = 0.73
    pos_adjust: float = 0.27
    pos_max: float = 95.0
    pos_floor: Dict[str, float] = field(default_factory=lambda: {
        "USA": 45.0, "China": 18.0, "Russia": 6.0,
        "Eurozone": 25.0, "Japan": 15.0, "UK": 18.0,
        "India": 8.0, "Saudi": 10.0
    })

    # Strength
    strength_inertia: float = 0.93
    strength_pos_w: float = 0.07


# ============================================================
# ОСНОВНОЙ КЛАСС
# ============================================================

class EndogenousGFN:
    def __init__(self,
                 countries: List[str] = None,
                 params: ModelParams = None,
                 seed: int = 42):

        self.countries = countries or [
            "USA", "China", "Russia", "Eurozone",
            "Japan", "UK", "India", "Saudi"
        ]
        self.params = params or ModelParams()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, year: int = 2026):
        self.year = year

        # ----- Финансовые переменные -----
        self.position = {
            "USA": 85.1, "China": 32.4, "Russia": 14.5,
            "Eurozone": 45.7, "Japan": 24.8, "UK": 29.8,
            "India": 11.7, "Saudi": 20.2
        }

        self.strength = {
            "USA": 0.92, "China": 0.71, "Russia": 0.29,
            "Eurozone": 0.68, "Japan": 0.55, "UK": 0.58,
            "India": 0.38, "Saudi": 0.42
        }

        self.confidence = {
            "USA": 0.76, "China": 0.64, "Russia": 0.42,
            "Eurozone": 0.61, "Japan": 0.58, "UK": 0.63,
            "India": 0.55, "Saudi": 0.57
        }

        self.leverage = {
            "USA": 0.48, "China": 0.57, "Russia": 0.65,
            "Eurozone": 0.52, "Japan": 0.55, "UK": 0.51,
            "India": 0.49, "Saudi": 0.44
        }

        self.npl = {
            "USA": 0.028, "China": 0.038, "Russia": 0.058,
            "Eurozone": 0.032, "Japan": 0.025, "UK": 0.030,
            "India": 0.045, "Saudi": 0.035
        }

        self.isolation = {
            "USA": 0.026, "China": 0.249, "Russia": 0.634,
            "Eurozone": 0.182, "Japan": 0.374, "UK": 0.257,
            "India": 0.462, "Saudi": 0.496
        }

        # ----- Технологический блок -----
        self.tech_centrality = {
            "USA": 0.92,
            "China": 0.48,
            "Russia": 0.18,
            "Eurozone": 0.55,
            "Japan": 0.62,
            "UK": 0.35,
            "India": 0.22,
            "Saudi": 0.12
        }

        self.tech_dependence = {
            "USA": 0.25,
            "China": 0.72,
            "Russia": 0.78,
            "Eurozone": 0.30,
            "Japan": 0.28,
            "UK": 0.35,
            "India": 0.65,
            "Saudi": 0.70
        }

        # tech_lag: технологическое отставание от фронтира
        frontier0 = max(self.tech_centrality[fc] for fc in self.params.frontier_countries)
        self.tech_lag = {
            c: max(0.0, frontier0 - self.tech_centrality[c])
            for c in self.countries
        }

        # Матрица финансовых связей
        link_data = {
            ("USA", "China"): 0.68, ("USA", "Russia"): 0.12,
            ("USA", "Eurozone"): 0.78, ("USA", "Japan"): 0.65,
            ("USA", "UK"): 0.81, ("USA", "India"): 0.42,
            ("USA", "Saudi"): 0.51,
            ("China", "Russia"): 0.48, ("China", "Eurozone"): 0.41,
            ("China", "Japan"): 0.37, ("China", "UK"): 0.33,
            ("China", "India"): 0.46, ("China", "Saudi"): 0.39,
            ("Russia", "Eurozone"): 0.19, ("Russia", "Japan"): 0.11,
            ("Russia", "UK"): 0.13, ("Russia", "India"): 0.28,
            ("Russia", "Saudi"): 0.31,
            ("Eurozone", "Japan"): 0.52, ("Eurozone", "UK"): 0.74,
            ("Eurozone", "India"): 0.36, ("Eurozone", "Saudi"): 0.34,
            ("Japan", "UK"): 0.47, ("Japan", "India"): 0.29,
            ("Japan", "Saudi"): 0.22,
            ("UK", "India"): 0.35, ("UK", "Saudi"): 0.29,
            ("India", "Saudi"): 0.33,
        }
        self.links = {frozenset(k): v for k, v in link_data.items()}

        self.history = []
        self._record()

    # ----------------------------------------------------------
    def _get_link(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.links.get(frozenset((a, b)), 0.15)

    def _set_link(self, a: str, b: str, val: float):
        p = self.params
        self.links[frozenset((a, b))] = float(np.clip(val, p.link_min, p.link_max))

    def _clip(self, val, lo, hi):
        return float(np.clip(val, lo, hi))

    def _is_sanctioned_pair(self, a: str, b: str) -> bool:
        return "Russia" in (a, b) or ("China" in (a, b) and "USA" in (a, b))

    # ----------------------------------------------------------
    def _evolve_links(self, sanctions: float, tech_sanctions: float):
        p = self.params
        n = len(self.countries)
        combined_pressure = max(sanctions, tech_sanctions * 0.7)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.countries[i], self.countries[j]
                current = self._get_link(a, b)

                fit_a = (p.fit_strength_w * self.strength[a] +
                         p.fit_confidence_w * self.confidence[a] +
                         p.fit_position_w * (self.position[a] / 100.0))
                fit_b = (p.fit_strength_w * self.strength[b] +
                         p.fit_confidence_w * self.confidence[b] +
                         p.fit_position_w * (self.position[b] / 100.0))

                desired = 0.5 * (fit_a + fit_b)

                if self._is_sanctioned_pair(a, b):
                    desired *= (1.0 - p.sanctions_desired_mult * combined_pressure)
                    current *= (1.0 - p.sanctions_current_mult * combined_pressure)

                new_link = (p.link_inertia * current + p.link_adjust * desired) * p.link_decay
                self._set_link(a, b, new_link)

    # ----------------------------------------------------------
    def _update_isolation(self):
        p = self.params
        for c in self.countries:
            others = [x for x in self.countries if x != c]
            avg_link = np.mean([self._get_link(c, o) for o in others])

            if c == "Russia":
                normal = p.isol_normal_russia
            elif c in ("USA", "Eurozone", "UK", "Japan"):
                normal = p.isol_normal_core
            else:
                normal = p.isol_normal_periphery

            isol = (normal - avg_link) / max(normal, 1e-6) * p.isol_scale
            self.isolation[c] = self._clip(isol, p.isol_min, p.isol_max)

    # ----------------------------------------------------------
    def _update_financial_health(self, shock: float, tech_sanctions: float):
        p = self.params
        for c in self.countries:
            isol = self.isolation[c]
            tech_dep = self.tech_dependence[c]

            self.leverage[c] = self._clip(
                self.leverage[c] + p.lev_base_growth + p.lev_isol_coef * isol + p.lev_shock_coef * shock,
                p.lev_min, p.lev_max)

            target_npl = ((self.leverage[c] - 0.5) * 0.3 + 0.2 * shock + 0.1 * isol)
            self.npl[c] = self._clip(
                self.npl[c] + p.npl_adjust_speed * (target_npl - self.npl[c]),
                p.npl_min, p.npl_max)

            # Confidence: tech sanctions + technological lag
            tech_hit = p.conf_tech_coef * tech_sanctions * tech_dep
            lag_hit = p.tech_lag_conf_coef * self.tech_lag.get(c, 0.0)
            self.confidence[c] = self._clip(
                self.confidence[c]
                - p.conf_shock_coef * shock
                - p.conf_isol_coef * isol
                - p.conf_npl_coef * (self.npl[c] - 0.04)
                - tech_hit
                - lag_hit,
                p.conf_min, p.conf_max)

    # ----------------------------------------------------------
    def _update_position(self, shock: float, tech_sanctions: float):
        p = self.params
        for c in self.countries:
            isol = self.isolation[c]
            tech_c = self.tech_centrality[c]
            tech_d = self.tech_dependence[c]

            isol_pen = p.isol_pen_coef * (isol - 0.1) * p.isol_pen_scale
            lev_pen  = p.lev_pen_coef * (self.leverage[c] - 0.45) * p.lev_pen_scale
            conf_b   = (self.confidence[c] - 0.5) * p.conf_bonus_scale
            residual = self.strength[c] * p.residual_scale * (1.0 - p.residual_isol_mult * isol)
            casc_hit = p.cascade_base * shock * (1.0 + p.cascade_isol_mult * isol)

            # Технологический эффект + штраф за отставание
            tech_effect = (p.tech_centrality_bonus * tech_c -
                           p.tech_dependence_penalty * tech_d * tech_sanctions)
            lag_pen = p.tech_lag_penalty * self.tech_lag.get(c, 0.0)

            target = (p.base_target - isol_pen - lev_pen + conf_b +
                      residual - casc_hit + tech_effect - lag_pen)

            if c == "USA":
                target = (1 - p.usa_anchor_w) * target + p.usa_anchor_w * p.usa_anchor
            elif c == "China":
                target = (1 - p.china_anchor_w) * target + p.china_anchor_w * p.china_anchor
            elif c == "Eurozone":
                target = (1 - p.euro_anchor_w) * target + p.euro_anchor_w * p.euro_anchor

            floor = p.pos_floor.get(c, 10.0)
            self.position[c] = self._clip(
                p.pos_inertia * self.position[c] + p.pos_adjust * target,
                floor, p.pos_max)

            self.strength[c] = (p.strength_inertia * self.strength[c] +
                                p.strength_pos_w * (self.position[c] / 100.0))

    # ----------------------------------------------------------
    def _update_tech(self, tech_sanctions: float):
        """
        Эндогенная динамика tech-переменных + технологическое отставание.

        tech_lag[c] = max(0, frontier - tech_centrality[c])
        frontier = max centrality среди USA / Japan / Eurozone

        - tech_centrality растёт с strength + catch-up (если isolation невысокая)
        - tech_sanctions создают drag (сильнее при высокой dependence)
        - tech_dependence снижается (import substitution), sanctions замедляют
        """
        p = self.params

        frontier = max(self.tech_centrality[fc] for fc in p.frontier_countries)

        for c in self.countries:
            strength = self.strength[c]
            dep = self.tech_dependence[c]
            isol = self.isolation[c]
            lag = max(0.0, frontier - self.tech_centrality[c])

            # --- Centrality ---
            growth = p.tech_c_growth * strength
            catchup = p.catchup_rate * lag * (1.0 - p.lag_isolation_mult * isol)
            drag = p.tech_c_sanction_drag * tech_sanctions * (0.4 + 0.6 * dep)

            new_c = self.tech_centrality[c] + growth + catchup - drag
            self.tech_centrality[c] = self._clip(new_c, p.tech_c_min, p.tech_c_max)

            # --- Dependence ---
            reduction = p.tech_d_reduction * (1.0 - p.tech_d_sanction_slow * tech_sanctions)
            reduction *= (0.7 + 0.3 * self.tech_centrality[c])
            new_d = self.tech_dependence[c] - reduction
            self.tech_dependence[c] = self._clip(new_d, p.tech_d_min, p.tech_d_max)

            # обновляем lag
            if not hasattr(self, "tech_lag"):
                self.tech_lag = {}
            self.tech_lag[c] = max(0.0, frontier - self.tech_centrality[c])

    # ----------------------------------------------------------
    def step(self, shock: float = 0.04, sanctions: float = 0.55,
             tech_sanctions: float = 0.30):
        """
        tech_sanctions: 0–1, интенсивность технологических ограничений
        (Entity List, export controls, CHIPS Act и т.д.)
        """
        self._evolve_links(sanctions, tech_sanctions)
        self._update_isolation()
        self._update_financial_health(shock, tech_sanctions)
        self._update_position(shock, tech_sanctions)
        self._update_tech(tech_sanctions)
        self.year += 1
        self._record()

    def _record(self):
        row = {"year": self.year}
        for c in self.countries:
            row[f"pos_{c}"] = round(self.position[c], 2)
            row[f"isol_{c}"] = round(self.isolation[c], 3)
            row[f"conf_{c}"] = round(self.confidence[c], 3)
            row[f"tech_c_{c}"] = round(self.tech_centrality[c], 3)
            row[f"tech_d_{c}"] = round(self.tech_dependence[c], 3)
            row[f"tech_lag_{c}"] = round(self.tech_lag.get(c, 0.0), 3)

        key_pairs = [
            ("USA", "China"), ("USA", "Russia"), ("China", "Russia"),
            ("USA", "Eurozone"), ("China", "Eurozone")
        ]
        for a, b in key_pairs:
            row[f"link_{a}_{b}"] = round(self._get_link(a, b), 3)

        self.history.append(row)

    # ----------------------------------------------------------
    def run(self, years: int = 10,
            shock_sched: Optional[List[float]] = None,
            sanc_sched: Optional[List[float]] = None,
            tech_sched: Optional[List[float]] = None,
            random_shock: bool = True) -> pd.DataFrame:

        if shock_sched is None:
            shock_sched = [0.04] * years
        if sanc_sched is None:
            sanc_sched = [0.55] * years
        if tech_sched is None:
            tech_sched = [0.30] * years

        for t in range(years):
            shock = shock_sched[t] if t < len(shock_sched) else 0.04
            sanc = sanc_sched[t] if t < len(sanc_sched) else 0.55
            tech = tech_sched[t] if t < len(tech_sched) else 0.30

            if random_shock and self.rng.random() < 0.12:
                shock += abs(self.rng.normal(0.03, 0.02))

            self.step(max(0.0, shock),
                      float(np.clip(sanc, 0.0, 1.0)),
                      float(np.clip(tech, 0.0, 1.0)))

        return pd.DataFrame(self.history)

    # ----------------------------------------------------------
    # Сценарии
    # ----------------------------------------------------------
    def scenario_baseline(self, years: int = 10) -> pd.DataFrame:
        self.reset()
        return self.run(years=years)

    def scenario_tech_escalation(self, years: int = 10) -> pd.DataFrame:
        """Усиление технологических ограничений против Китая"""
        self.reset()
        shock = [0.04] * years
        sanc = [0.50] * years
        tech = np.linspace(0.35, 0.85, years).tolist()
        return self.run(years=years, shock_sched=shock,
                        sanc_sched=sanc, tech_sched=tech, random_shock=False)

    def scenario_tech_lag(self, years: int = 10) -> pd.DataFrame:
        """
        Сценарий технологического отставания:
        сильные и растущие tech_sanctions + умеренные финансовые санкции.
        Catch-up затруднён, lag закрепляется / растёт у зависимых стран.
        """
        self.reset()
        shock = [0.05] * years
        sanc = np.linspace(0.45, 0.70, years).tolist()
        tech = np.linspace(0.50, 0.95, years).tolist()
        return self.run(years=years, shock_sched=shock,
                        sanc_sched=sanc, tech_sched=tech, random_shock=False)

    def scenario_dual_decoupling(self, years: int = 10) -> pd.DataFrame:
        """Одновременное финансовое и технологическое давление"""
        self.reset()
        shock = np.linspace(0.05, 0.12, years).tolist()
        sanc = np.linspace(0.55, 0.88, years).tolist()
        tech = np.linspace(0.40, 0.90, years).tolist()
        return self.run(years=years, shock_sched=shock,
                        sanc_sched=sanc, tech_sched=tech, random_shock=False)

    def scenario_russia_focus(self, years: int = 12) -> pd.DataFrame:
        self.reset()
        shock = [0.04]*3 + [0.11, 0.09, 0.07] + [0.05]*(years-6)
        sanc = [0.50]*3 + [0.88, 0.85, 0.80] + [0.72]*(years-6)
        tech = [0.25]*3 + [0.45, 0.50, 0.55] + [0.50]*(years-6)
        return self.run(years=years, shock_sched=shock,
                        sanc_sched=sanc, tech_sched=tech, random_shock=False)

    def get_link_matrix(self) -> pd.DataFrame:
        n = len(self.countries)
        mat = np.zeros((n, n))
        for i, a in enumerate(self.countries):
            for j, b in enumerate(self.countries):
                if i != j:
                    mat[i, j] = self._get_link(a, b)
        return pd.DataFrame(mat, index=self.countries, columns=self.countries)


# ============================================================
# ДЕМОНСТРАЦИЯ
# ============================================================

if __name__ == "__main__":
    print("=" * 78)
    print("EndogenousGFN + Technology Block + Tech Lag")
    print("=" * 78)

    cols = ["year", "pos_USA", "pos_China", "pos_Russia",
            "tech_lag_China", "tech_lag_Russia"]

    print("\n>>> Baseline (6 years)")
    model = EndogenousGFN(seed=42)
    df = model.scenario_baseline(6)
    print(df[[c for c in cols if c in df.columns]].round(2).to_string(index=False))

    print("\n>>> Tech Escalation scenario")
    model2 = EndogenousGFN(seed=42)
    df_tech = model2.scenario_tech_escalation(6)
    print(df_tech[[c for c in cols if c in df_tech.columns]].round(2).to_string(index=False))

    print("\n>>> Tech Lag scenario")
    model3 = EndogenousGFN(seed=42)
    df_lag = model3.scenario_tech_lag(6)
    print(df_lag[[c for c in cols if c in df_lag.columns]].round(2).to_string(index=False))

    print("\n>>> Dual Decoupling scenario")
    model4 = EndogenousGFN(seed=42)
    df_dual = model4.scenario_dual_decoupling(6)
    print(df_dual[[c for c in cols if c in df_dual.columns]].round(2).to_string(index=False))