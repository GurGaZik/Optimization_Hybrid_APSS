from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from time import perf_counter
from typing import Callable, Literal

import numpy as np

from data_loader import SourceData
from energy_models import EquipmentChoice, SimulationResult, SystemConfig, simulate_system

AlgorithmName = Literal["Greedy", "GWO", "DE"]


@dataclass
class OptimizationConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    random_seed: int = 42

    de_maxiter: int = 5
    de_popsize: int = 3
    gwo_epoch: int = 8
    gwo_pop_size: int = 8

    # Веса целевой функции. Порядок значимости: топливо -> использование ВЭУ -> цена.
    unserved_penalty: float = 1e9
    invalid_penalty: float = 1e8
    wind_used_reward: float = 0.005
    wind_curtail_penalty: float = 0.001
    equipment_cost_scale: float = 1e6
    diesel_high_load_penalty: float = 0.01
    ballast_penalty: float = 0.002


@dataclass
class OptimizationResult:
    algorithm: AlgorithmName
    best: SimulationResult
    objective_value: float
    elapsed_seconds: float
    evaluations: int

    def report(self) -> str:
        return (
            f"Алгоритм: {self.algorithm}\n"
            f"Значение целевой функции: {self.objective_value:.6f}\n"
            f"Количество расчетов режима: {self.evaluations}\n"
            f"Время расчета: {self.elapsed_seconds:.2f} с\n\n"
            f"{self.best.short_report()}"
        )


class ObjectiveEvaluator:
    def __init__(self, data: SourceData, config: OptimizationConfig):
        self.data = data
        self.config = config
        self.evaluations = 0
        self.best_result: SimulationResult | None = None
        self.best_value = float("inf")
        self._cache: dict[tuple[int, int, int, float, int, int], tuple[float, SimulationResult]] = {}

    def score_result(self, result: SimulationResult) -> float:
        m = result.metrics
        equipment = result.equipment
        score = float(m["diesel_fuel_l"])
        score += self.config.unserved_penalty * float(m["unserved_kwh"])
        if not equipment.valid:
            score += self.config.invalid_penalty
        score -= self.config.wind_used_reward * float(m["wind_used_kwh"])
        score += self.config.wind_curtail_penalty * float(m["wind_curtailed_kwh"])
        score += float(m["equipment_cost_rub"]) / self.config.equipment_cost_scale
        score += self.config.diesel_high_load_penalty * float(m["diesel_high_load_kwh"])
        score += self.config.ballast_penalty * float(m.get("ballast_kwh", 0.0))
        return score

    def evaluate_choice(self, choice: EquipmentChoice) -> tuple[float, SimulationResult]:
        key = (
            int(choice.wind_index),
            int(choice.wind_count),
            int(choice.battery_index),
            round(float(choice.battery_target_kwh), 3),
            int(choice.inverter_index),
            int(choice.inverter_count),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = simulate_system(self.data, choice, self.config.system, make_hourly=False)
        score = self.score_result(result)
        self.evaluations += 1
        self._cache[key] = (score, result)
        if score < self.best_value:
            self.best_value = score
            self.best_result = result
        return score, result

    def vector_to_choice(self, x: np.ndarray) -> EquipmentChoice:
        x = np.asarray(x, dtype=float)
        wt_n = len(self.data.wind_turbines)
        bat_n = len(self.data.batteries)
        inv_n = len(self.data.inverters)
        max_wind = self.config.system.max_wind_turbines
        max_inv = _estimate_max_inverters(self.data, self.config.system)
        max_batt = _estimate_max_battery_kwh(self.data, self.config.system)

        return EquipmentChoice(
            wind_index=int(np.clip(round(x[0]), 0, wt_n - 1)),
            wind_count=int(np.clip(round(x[1]), 0, max_wind)),
            battery_index=int(np.clip(round(x[2]), 0, bat_n - 1)),
            battery_target_kwh=float(np.clip(x[3], 0.0, max_batt)),
            inverter_index=int(np.clip(round(x[4]), 0, inv_n - 1)),
            inverter_count=int(np.clip(round(x[5]), 0, max_inv)),
        )

    def evaluate_vector(self, x: np.ndarray) -> float:
        choice = self.vector_to_choice(x)
        score, _ = self.evaluate_choice(choice)
        return score


def run_optimization(data: SourceData, algorithm: AlgorithmName = "Greedy", config: OptimizationConfig | None = None) -> OptimizationResult:
    config = config or OptimizationConfig()
    algorithm = _normalize_algorithm_name(algorithm)
    start = perf_counter()
    evaluator = ObjectiveEvaluator(data, config)

    if algorithm == "Greedy":
        best_score, best_result = _run_greedy(data, evaluator, config)
    elif algorithm == "DE":
        best_score, best_result = _run_de(data, evaluator, config)
    elif algorithm == "GWO":
        best_score, best_result = _run_gwo(data, evaluator, config)
    else:
        raise ValueError(f"Неизвестный алгоритм оптимизации: {algorithm}")

    best_result = simulate_system(data, best_result.choice, config.system, make_hourly=True)
    best_score = evaluator.score_result(best_result)

    elapsed = perf_counter() - start
    return OptimizationResult(
        algorithm=algorithm,
        best=best_result,
        objective_value=best_score,
        elapsed_seconds=elapsed,
        evaluations=evaluator.evaluations,
    )


def _run_greedy(data: SourceData, evaluator: ObjectiveEvaluator, config: OptimizationConfig) -> tuple[float, SimulationResult]:
    """
    Быстрый жадный поиск.
    1. Подбирает модель и количество ВЭУ.
    2. Для найденной ветровой части подбирает АКБ и инверторы.
    """
    system = config.system
    base = EquipmentChoice(wind_index=0, wind_count=0, battery_index=0, battery_target_kwh=0.0, inverter_index=0, inverter_count=0)
    best_score, best_result = evaluator.evaluate_choice(base)

    # Шаг 1: ВЭУ. Greedy использует короткую сетку, чтобы быстро дать базовое решение.
    wind_pool = _top_wind_indices(data, limit=6)
    wind_count_levels = _unique_int_levels([0, 2, 4, 8, 12, 14, 16, 18, system.max_wind_turbines], 0, system.max_wind_turbines)
    for w_idx in wind_pool:
        for w_count in wind_count_levels:
            choice = EquipmentChoice(w_idx, w_count, 0, 0.0, 0, 0)
            score, result = evaluator.evaluate_choice(choice)
            if score < best_score:
                best_score, best_result = score, result

    best_w_idx = best_result.choice.wind_index
    best_w_count = best_result.choice.wind_count

    # Шаг 2: АКБ и инверторы.
    max_batt = _estimate_max_battery_kwh(data, system)
    max_inv = _estimate_max_inverters(data, system)
    peak_deficit = _peak_deficit_after_mges(data)
    battery_pool = _top_battery_indices(data, limit=4)
    inverter_pool = _top_inverter_indices(data, limit=4)

    for inv_idx in inverter_pool:
        inv = data.inverters.reset_index(drop=True).iloc[inv_idx]
        rated = float(inv.get("rated_power_kw", 0.0))
        if rated <= 0:
            continue
        inv_count_levels = _unique_int_levels([
            0,
            ceil(0.75 * peak_deficit / rated),
            ceil(1.25 * peak_deficit / rated),
        ], 0, max_inv)
        for inv_count in inv_count_levels:
            inv_power = inv_count * rated
            battery_levels = _unique_float_levels([
                0.0,
                1.5 * inv_power,
                4.0 * inv_power,
                max_batt,
            ], 0.0, max_batt)
            for bat_idx in battery_pool:
                for target_kwh in battery_levels:
                    choice = EquipmentChoice(best_w_idx, best_w_count, bat_idx, target_kwh, inv_idx, inv_count)
                    score, result = evaluator.evaluate_choice(choice)
                    if score < best_score:
                        best_score, best_result = score, result

    return best_score, best_result


def _run_de(data: SourceData, evaluator: ObjectiveEvaluator, config: OptimizationConfig) -> tuple[float, SimulationResult]:
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError("Для алгоритма DE установите библиотеку scipy: pip install scipy") from exc

    bounds = _continuous_bounds(data, config.system)
    differential_evolution(
        evaluator.evaluate_vector,
        bounds=bounds,
        seed=config.random_seed,
        maxiter=config.de_maxiter,
        popsize=config.de_popsize,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.01,
    )
    if evaluator.best_result is None:
        raise RuntimeError("DE не вернул ни одного допустимого результата.")
    return evaluator.best_value, evaluator.best_result


def _run_gwo(data: SourceData, evaluator: ObjectiveEvaluator, config: OptimizationConfig) -> tuple[float, SimulationResult]:
    try:
        from mealpy import FloatVar
        from mealpy.swarm_based.GWO import OriginalGWO
    except ImportError as exc:
        raise RuntimeError("Для алгоритма GWO установите библиотеку mealpy: pip install mealpy") from exc

    bounds = _continuous_bounds(data, config.system)
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    problem = {
        "bounds": FloatVar(lb=lb, ub=ub, name="equipment"),
        "minmax": "min",
        "obj_func": evaluator.evaluate_vector,
    }
    model = OriginalGWO(epoch=config.gwo_epoch, pop_size=config.gwo_pop_size)
    model.solve(problem, seed=config.random_seed)

    if evaluator.best_result is None:
        raise RuntimeError("GWO не вернул ни одного допустимого результата.")
    return evaluator.best_value, evaluator.best_result



def _top_wind_indices(data: SourceData, limit: int) -> list[int]:
    df = data.wind_turbines.reset_index(drop=True).copy()
    df["_score"] = df["price_rub"] / df["rated_power_kw"].replace(0, np.nan)
    return df.sort_values("_score").head(min(limit, len(df))).index.astype(int).tolist()


def _top_battery_indices(data: SourceData, limit: int) -> list[int]:
    df = data.batteries.reset_index(drop=True).copy()
    df["_score"] = df["price_rub"] / df["energy_kwh"].replace(0, np.nan)
    return df.sort_values("_score").head(min(limit, len(df))).index.astype(int).tolist()


def _top_inverter_indices(data: SourceData, limit: int) -> list[int]:
    df = data.inverters.reset_index(drop=True).copy()
    df["_score"] = df["price_rub"] / df["rated_power_kw"].replace(0, np.nan)
    return df.sort_values("_score").head(min(limit, len(df))).index.astype(int).tolist()


def _continuous_bounds(data: SourceData, system: SystemConfig) -> list[tuple[float, float]]:
    return [
        (0.0, max(0.0, len(data.wind_turbines) - 1.0)),
        (0.0, float(system.max_wind_turbines)),
        (0.0, max(0.0, len(data.batteries) - 1.0)),
        (0.0, _estimate_max_battery_kwh(data, system)),
        (0.0, max(0.0, len(data.inverters) - 1.0)),
        (0.0, float(_estimate_max_inverters(data, system))),
    ]


def _estimate_max_battery_kwh(data: SourceData, system: SystemConfig) -> float:
    if system.max_battery_target_kwh is not None:
        return float(system.max_battery_target_kwh)
    peak_deficit = _peak_deficit_after_mges(data)
    return float(max(500.0, peak_deficit * 4.0))


def _estimate_max_inverters(data: SourceData, system: SystemConfig) -> int:
    min_rated = float(data.inverters["rated_power_kw"].replace(0, np.nan).min())
    peak_deficit = _peak_deficit_after_mges(data)
    if not np.isfinite(min_rated) or min_rated <= 0:
        return min(system.max_inverters, 100)
    return int(min(system.max_inverters, max(1, ceil(1.25 * peak_deficit / min_rated))))


def _peak_deficit_after_mges(data: SourceData) -> float:
    residual = data.load_hour["load_kwh"].to_numpy(dtype=float) - data.mges_hour["mges_total_kwh"].to_numpy(dtype=float)
    return float(max(0.0, np.max(residual)))


def _unique_int_levels(values: list[int], low: int, high: int) -> list[int]:
    return sorted({int(min(max(v, low), high)) for v in values})


def _unique_float_levels(values: list[float], low: float, high: float) -> list[float]:
    result = []
    for v in values:
        vv = float(min(max(v, low), high))
        if all(abs(vv - old) > 1e-9 for old in result):
            result.append(vv)
    return sorted(result)


def _normalize_algorithm_name(algorithm: str) -> AlgorithmName:
    text = str(algorithm).strip().upper()
    if text in {"GREEDY", "ЖАДНЫЙ"}:
        return "Greedy"
    if text == "GWO":
        return "GWO"
    if text == "DE":
        return "DE"
    raise ValueError(f"Неизвестный алгоритм оптимизации: {algorithm}")


if __name__ == "__main__":
    from data_loader import load_source_data

    data = load_source_data("Исходные данные.xlsx")
    result = run_optimization(data, algorithm="Greedy")
    print(result.report())
