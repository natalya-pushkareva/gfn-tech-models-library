# GFN & Technology Models Library (2025–2035)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22766267.svg)](https://doi.org/10.5281/zenodo.22766267)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Библиотека из четырёх взаимосвязанных вычислительных моделей глобальной финансовой сети (GFN), каскадных рисков, технологического индекса и технологического отставания с акцентом на Россию, США и Китай (горизонт 2025–2035).

## Модели

| Модель | Файл | Краткое описание |
|--------|------|------------------|
| **Технологический Индекс 2035** | `tech_index_2035.py` | Monte-Carlo + Numba. Гибрид SD + ABM. Сценарии Россия / Китай / США. |
| **RUS-GFN-TSI-2035 CASCADE** | `rus_gfn_tsi_cascade.py` | Каскадное распространение дефолтов (Gai–Kapadia) + SDM России + GFN. |
| **GFN+SBS Twin-Simulation** | `gfn_sbs_twin.py` | Twin-симуляция (bailout vs pure market) с калибровкой на 2026. SBS + изгнание РФ. |
| **EndogenousGFN + Tech Lag** | `endogenous_gfn_tech.py` | Эндогенная эволюция GFN + tech_centrality / dependence / lag + сценарии санкций. |

## Установка

```bash
conda env create -f environment.yml
conda activate gfn-tech-models
# или
pip install -e .