from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from data_loader import SourceData


@dataclass(frozen=True)
class SystemConfig:
    # ДЭС: 2 x Caterpillar 3516B по 1,6 МВт
    diesel_unit_power_kw: float = 1600.0
    diesel_units_count: int = 2
    diesel_min_load_fraction: float = 0.40
    diesel_preferred_max_load_fraction: float = 0.80

    diesel_fuel_load_points: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    diesel_fuel_lph_points: tuple[float, ...] = (122.9, 213.8, 305.3, 409.1)

    # ВЭУ
    wind_reference_height_m: float = 10.0
    wind_profile_alpha: float = 0.14
    wind_cut_in_mps: float = 3.0
    wind_rated_mps: float = 12.0
    wind_cut_out_mps: float = 25.0
    air_density_kg_m3: float = 1.225
    wind_a_induction: float = 1.0 / 3.0
    wind_mech_elec_eta: float = 0.90

    # АКБ/инвертор
    battery_dod: float = 0.80
    battery_initial_soc: float = 0.50
    battery_eta_charge: float = 0.96
    battery_eta_discharge: float = 0.96
    inverter_eta: float = 0.95
    grid_voltage_v: float = 380.0

    # Pуст_ВЭУ <= min(Pостаточная после МГЭС, PДЭС_уст).
    max_wind_turbines: int = 20
    wind_installed_power_limit_fraction: float = 1.00
    max_inverters: int = 300
    max_battery_target_kwh: float | None = None


@dataclass(frozen=True)
class EquipmentChoice:
    wind_index: int
    wind_count: int
    battery_index: int
    battery_target_kwh: float
    inverter_index: int
    inverter_count: int


@dataclass
class ResolvedEquipment:
    wind: dict[str, Any]
    battery: dict[str, Any]
    inverter: dict[str, Any]
    valid: bool
    violations: list[str] = field(default_factory=list)

    @property
    def total_price_rub(self) -> float:
        return float(self.wind.get("price_total_rub", 0.0) + self.battery.get("price_total_rub", 0.0) + self.inverter.get("price_total_rub", 0.0))


@dataclass
class SimulationResult:
    choice: EquipmentChoice
    equipment: ResolvedEquipment
    hourly: pd.DataFrame
    metrics: dict[str, float]

    def short_report(self) -> str:
        e = self.equipment
        m = self.metrics
        lines = [
            "Выбранное оборудование:",
            f"  ВЭУ: {e.wind.get('model', '-')} x {e.wind.get('count', 0)} шт., Pуст = {e.wind.get('installed_power_kw', 0):.1f} кВт",
            f"  АКБ: {e.battery.get('model', '-')} x {e.battery.get('total_cells', 0)} элементов; схема {e.battery.get('series_count', 0)}s x {e.battery.get('parallel_count', 0)}p; E = {e.battery.get('energy_total_kwh', 0):.1f} кВт·ч",
            f"  Инвертор: {e.inverter.get('model', '-')} x {e.inverter.get('count', 0)} шт., Pном = {e.inverter.get('rated_power_total_kw', 0):.1f} кВт",
            f"  Стоимость оборудования: {e.total_price_rub:,.0f} руб".replace(",", " "),
            "Результаты режима:",
            f"  Расход топлива ДЭС: {m['diesel_fuel_l']:.1f} л",
            f"  Выработка ДЭС: {m['diesel_energy_kwh']:.1f} кВт·ч",
            f"  Использовано ВЭУ: {m['wind_used_kwh']:.1f} кВт·ч из {m['wind_available_kwh']:.1f} кВт·ч (Kиспл = {m['wind_utilization_fraction'] * 100:.2f} %)",
            f"  Непокрытая нагрузка: {m['unserved_kwh']:.1f} кВт·ч",
            f"  Балластная нагрузка: {m.get('ballast_kwh', 0.0):.1f} кВт·ч",
        ]
        if e.violations:
            lines.append("Ограничения/предупреждения:")
            lines.extend(f"  - {v}" for v in e.violations)
        return "\n".join(lines)


def peak_load_kw(data: SourceData) -> float:
    return float(np.max(data.load_hour["load_kwh"].to_numpy(dtype=float)))


def peak_residual_after_mges_kw(data: SourceData) -> float:
    load = data.load_hour["load_kwh"].to_numpy(dtype=float)
    mges = data.mges_hour["mges_total_kwh"].to_numpy(dtype=float)
    return float(max(0.0, np.max(load - mges)))


def wind_installed_power_limit_kw(data: SourceData, config: SystemConfig) -> float:

    residual_peak = peak_residual_after_mges_kw(data)
    diesel_installed = config.diesel_unit_power_kw * config.diesel_units_count
    base_limit = min(residual_peak, diesel_installed)
    return float(max(0.0, base_limit * config.wind_installed_power_limit_fraction))


def max_wind_count_for_turbine(data: SourceData, turbine: pd.Series | dict[str, Any], config: SystemConfig) -> int:
    rated_power = float(turbine.get("rated_power_kw", 0.0))
    if rated_power <= 0:
        return 0
    by_power = int(np.floor(wind_installed_power_limit_kw(data, config) / rated_power))
    return int(max(0, min(config.max_wind_turbines, by_power)))


def cp_from_axial_induction(a: float) -> float:

    return float(4.0 * a * (1.0 - a) ** 2)


def wind_speed_at_hub(v10_mps: np.ndarray, hub_height_m: float, config: SystemConfig) -> np.ndarray:
    return np.asarray(v10_mps, dtype=float) * (hub_height_m / config.wind_reference_height_m) ** config.wind_profile_alpha


def _power_curve_arrays_from_turbine(turbine: pd.Series | dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    speeds_raw = turbine.get("power_curve_speeds_mps", tuple())
    powers_raw = turbine.get("power_curve_powers_kw", tuple())

    if speeds_raw is None or powers_raw is None:
        return None

    try:
        speeds = np.asarray(list(speeds_raw), dtype=float)
        powers = np.asarray(list(powers_raw), dtype=float)
    except Exception:
        return None

    if speeds.shape != powers.shape:
        return None

    mask = np.isfinite(speeds) & np.isfinite(powers)
    speeds = speeds[mask]
    powers = powers[mask]
    if len(speeds) < 2:
        return None

    order = np.argsort(speeds)
    speeds = speeds[order]
    powers = np.maximum(0.0, powers[order])

    unique_speeds: list[float] = []
    unique_powers: list[float] = []
    for speed, power in zip(speeds, powers):
        if unique_speeds and abs(float(speed) - unique_speeds[-1]) < 1e-9:
            unique_powers[-1] = float(power)
        else:
            unique_speeds.append(float(speed))
            unique_powers.append(float(power))

    if len(unique_speeds) < 2:
        return None
    return np.asarray(unique_speeds, dtype=float), np.asarray(unique_powers, dtype=float)


def wind_power_curve_kw(v_mps: np.ndarray, turbine: pd.Series | dict[str, Any], count: int, config: SystemConfig) -> np.ndarray:
    """
    Расчет мощности ВЭУ.

    Приоритет 1: паспортная кривая мощности с листа «ВЭУ» исходного Excel.
    Мощность при промежуточных скоростях определяется линейной интерполяцией.

    Приоритет 2: если паспортной кривой нет, используется расчетная модель:
    физическая формула через диаметр ротора или нормированная кусочная характеристика.
    """
    v = np.asarray(v_mps, dtype=float)
    count = int(max(0, count))
    if count == 0:
        return np.zeros_like(v, dtype=float)

    rated_power = float(turbine.get("rated_power_kw", 0.0))
    if rated_power <= 0:
        return np.zeros_like(v, dtype=float)

    curve = _power_curve_arrays_from_turbine(turbine)
    if curve is not None:
        speeds, powers = curve
        p_one = np.interp(v, speeds, powers, left=0.0, right=0.0)
        p_one = np.maximum(0.0, p_one)
        return p_one * count

    cut_in = float(turbine.get("cut_in_speed_mps", np.nan))
    rated_speed = float(turbine.get("rated_speed_mps", np.nan))
    cut_out = float(turbine.get("cut_out_speed_mps", np.nan))
    if not np.isfinite(cut_in) or cut_in <= 0:
        cut_in = config.wind_cut_in_mps
    if not np.isfinite(rated_speed) or rated_speed <= cut_in:
        rated_speed = config.wind_rated_mps
    if not np.isfinite(cut_out) or cut_out <= rated_speed:
        cut_out = config.wind_cut_out_mps

    rotor_diameter = float(turbine.get("rotor_diameter_m", np.nan))
    cp = cp_from_axial_induction(config.wind_a_induction)
    cp_max = 16.0 / 27.0

    if np.isfinite(rotor_diameter) and rotor_diameter > 0:
        swept_area = np.pi * (rotor_diameter / 2.0) ** 2
        p_one = 0.5 * config.air_density_kg_m3 * swept_area * np.power(v, 3) * cp * config.wind_mech_elec_eta / 1000.0
        p_one = np.clip(p_one, 0.0, rated_power)
        p_one = np.where((v >= cut_in) & (v <= cut_out), p_one, 0.0)
        return p_one * count

    p_one = np.zeros_like(v, dtype=float)
    mask_ramp = (v >= cut_in) & (v < rated_speed)
    denom = rated_speed**3 - cut_in**3
    p_one[mask_ramp] = rated_power * ((v[mask_ramp] ** 3 - cut_in**3) / denom) * (cp / cp_max)
    p_one[(v >= rated_speed) & (v <= cut_out)] = rated_power
    p_one = np.clip(p_one, 0.0, rated_power)
    return p_one * count


def fuel_lph_for_one_diesel(power_kw: float, config: SystemConfig) -> float:
    if power_kw <= 1e-9:
        return 0.0
    load_fraction = power_kw / config.diesel_unit_power_kw
    return float(np.interp(load_fraction, config.diesel_fuel_load_points, config.diesel_fuel_lph_points))


def choose_diesel_dispatch(required_kw: float, config: SystemConfig) -> dict[str, Any]:
    """Выбирает число ДЭУ и распределение мощности с учетом kз >= 0,4."""
    required_kw = float(max(0.0, required_kw))
    if required_kw <= 1e-9:
        return {"units": 0, "total_power_kw": 0.0, "per_unit_power_kw": 0.0, "fuel_lph": 0.0, "load_fraction": 0.0, "high_load_kw": 0.0, "unserved_kw": 0.0}

    min_unit = config.diesel_min_load_fraction * config.diesel_unit_power_kw
    max_unit = config.diesel_unit_power_kw
    preferred_unit = config.diesel_preferred_max_load_fraction * config.diesel_unit_power_kw

    candidates: list[dict[str, Any]] = []
    for units in range(1, config.diesel_units_count + 1):
        total_power = max(required_kw, units * min_unit)
        if total_power > units * max_unit + 1e-9:
            continue
        per_unit = total_power / units
        if per_unit < min_unit - 1e-9 or per_unit > max_unit + 1e-9:
            continue
        fuel = units * fuel_lph_for_one_diesel(per_unit, config)
        high_load_kw = max(0.0, per_unit - preferred_unit) * units
        # Топливо — главный критерий. Перегрузка свыше 80% допустима, но получает небольшой штраф выбора.
        score = fuel + 0.02 * high_load_kw
        candidates.append({
            "units": units,
            "total_power_kw": total_power,
            "per_unit_power_kw": per_unit,
            "fuel_lph": fuel,
            "load_fraction": per_unit / config.diesel_unit_power_kw,
            "high_load_kw": high_load_kw,
            "unserved_kw": 0.0,
            "score": score,
        })

    if candidates:
        return min(candidates, key=lambda x: x["score"])

    total_max = config.diesel_units_count * max_unit
    fuel = config.diesel_units_count * fuel_lph_for_one_diesel(max_unit, config)
    return {"units": config.diesel_units_count, "total_power_kw": total_max, "per_unit_power_kw": max_unit, "fuel_lph": fuel, "load_fraction": 1.0, "high_load_kw": total_max - config.diesel_units_count * preferred_unit, "unserved_kw": required_kw - total_max}


def resolve_equipment(data: SourceData, choice: EquipmentChoice, config: SystemConfig) -> ResolvedEquipment:
    wt_df = data.wind_turbines.reset_index(drop=True)
    bat_df = data.batteries.reset_index(drop=True)
    inv_df = data.inverters.reset_index(drop=True)

    w_idx = int(np.clip(choice.wind_index, 0, len(wt_df) - 1))
    b_idx = int(np.clip(choice.battery_index, 0, len(bat_df) - 1))
    i_idx = int(np.clip(choice.inverter_index, 0, len(inv_df) - 1))

    wt = wt_df.iloc[w_idx]
    bat = bat_df.iloc[b_idx]
    inv = inv_df.iloc[i_idx]

    wind_count_limit = max_wind_count_for_turbine(data, wt, config)
    wind_count = int(np.clip(round(choice.wind_count), 0, wind_count_limit))
    inverter_count = int(np.clip(round(choice.inverter_count), 0, config.max_inverters))
    battery_target_kwh = float(max(0.0, choice.battery_target_kwh))

    violations: list[str] = []
    valid = True

    wind = {
        "id": int(wt.get("id", w_idx + 1)),
        "model": str(wt.get("model", "-")),
        "count": wind_count,
        "rated_power_kw": float(wt.get("rated_power_kw", 0.0)),
        "hub_height_m": float(wt.get("hub_height_m", config.wind_reference_height_m)),
        "installed_power_kw": wind_count * float(wt.get("rated_power_kw", 0.0)),
        "count_limit": wind_count_limit,
        "installed_power_limit_kw": wind_installed_power_limit_kw(data, config),
        "price_total_rub": wind_count * float(wt.get("price_rub", 0.0)),
        "row": wt.to_dict(),
    }

    inv_rated = float(inv.get("rated_power_kw", 0.0))
    inv_max = float(inv.get("max_power_kw", inv_rated))
    inv_peak = float(inv.get("peak_power_kw", inv_max))
    inv_input_v = float(inv.get("input_voltage_v", 0.0))
    inv_output_v = float(inv.get("output_voltage_v", config.grid_voltage_v))
    inv_cmax_ah = float(inv.get("max_capacity_ah", np.inf))

    inverter = {
        "id": int(inv.get("id", i_idx + 1)),
        "model": str(inv.get("model", "-")) if inverter_count > 0 else "не выбран",
        "count": inverter_count,
        "rated_power_kw": inv_rated,
        "max_power_kw": inv_max,
        "peak_power_kw": inv_peak,
        "input_voltage_v": inv_input_v,
        "output_voltage_v": inv_output_v,
        "rated_power_total_kw": inverter_count * inv_rated,
        "max_power_total_kw": inverter_count * inv_max,
        "peak_power_total_kw": inverter_count * inv_peak,
        "max_capacity_total_ah": inverter_count * inv_cmax_ah,
        "price_total_rub": inverter_count * float(inv.get("price_rub", 0.0)),
        "row": inv.to_dict(),
    }

    battery_enabled = battery_target_kwh > 1e-9 and inverter_count > 0
    if battery_target_kwh > 1e-9 and inverter_count == 0:
        valid = False
        violations.append("Для выбранной АКБ не выбран инвертор.")

    if abs(inv_output_v - config.grid_voltage_v) > 1e-6 and inverter_count > 0:
        valid = False
        violations.append(f"Uвых инвертора {inv_output_v:g} В не соответствует напряжению сети {config.grid_voltage_v:g} В.")

    if battery_enabled:
        cell_voltage = float(bat.get("voltage_v", 0.0))
        cell_capacity_ah = float(bat.get("capacity_ah", 0.0))
        series_count = max(1, int(round(inv_input_v / cell_voltage))) if cell_voltage > 0 and inv_input_v > 0 else 0
        # Проверим соседние варианты последовательного соединения и выберем ближайшее напряжение.
        if cell_voltage > 0 and inv_input_v > 0:
            candidates = [max(1, series_count - 1), series_count, series_count + 1]
            series_count = min(candidates, key=lambda n: abs(n * cell_voltage - inv_input_v))
        pack_voltage = series_count * cell_voltage
        energy_per_parallel_kwh = pack_voltage * cell_capacity_ah / 1000.0
        parallel_count = max(1, int(ceil(battery_target_kwh / energy_per_parallel_kwh))) if energy_per_parallel_kwh > 0 else 0
        total_cells = series_count * parallel_count
        capacity_ah_total = parallel_count * cell_capacity_ah
        energy_total_kwh = pack_voltage * capacity_ah_total / 1000.0

        if capacity_ah_total > inverter["max_capacity_total_ah"] + 1e-9:
            valid = False
            violations.append(
                f"CАКБ={capacity_ah_total:.1f} А·ч больше допустимой Cmax инверторов={inverter['max_capacity_total_ah']:.1f} А·ч."
            )
    else:
        series_count = parallel_count = total_cells = 0
        pack_voltage = capacity_ah_total = energy_total_kwh = 0.0

    battery = {
        "id": int(bat.get("id", b_idx + 1)),
        "model": str(bat.get("model", "-")) if battery_enabled else "не выбрана",
        "cell_voltage_v": float(bat.get("voltage_v", 0.0)),
        "cell_capacity_ah": float(bat.get("capacity_ah", 0.0)),
        "series_count": int(series_count),
        "parallel_count": int(parallel_count),
        "total_cells": int(total_cells),
        "pack_voltage_v": float(pack_voltage),
        "capacity_ah": float(capacity_ah_total),
        "energy_total_kwh": float(energy_total_kwh),
        "energy_usable_kwh": float(energy_total_kwh * config.battery_dod),
        "price_total_rub": total_cells * float(bat.get("price_rub", 0.0)),
        "row": bat.to_dict(),
    }

    return ResolvedEquipment(wind=wind, battery=battery, inverter=inverter, valid=valid, violations=violations)


def simulate_system(
    data: SourceData,
    choice: EquipmentChoice,
    config: SystemConfig | None = None,
    make_hourly: bool = True,
) -> SimulationResult:
    config = config or SystemConfig()
    equipment = resolve_equipment(data, choice, config)

    load = data.load_hour["load_kwh"].to_numpy(dtype=float)
    mges1 = data.mges_hour["mges1_kwh"].to_numpy(dtype=float)
    mges2 = data.mges_hour["mges2_kwh"].to_numpy(dtype=float)
    mges = data.mges_hour["mges_total_kwh"].to_numpy(dtype=float)
    v10 = data.wind_hour["wind_speed_10m_mps"].to_numpy(dtype=float)
    datetimes = data.load_hour["datetime"].to_numpy() if make_hourly else None

    wt_row = equipment.wind["row"]
    v_hub = wind_speed_at_hub(v10, equipment.wind["hub_height_m"], config)
    wind_available = wind_power_curve_kw(v_hub, wt_row, equipment.wind["count"], config)

    n = len(load)
    soc_energy = equipment.battery["energy_total_kwh"] * config.battery_initial_soc
    soc_min = equipment.battery["energy_total_kwh"] * (1.0 - config.battery_dod)
    soc_max = equipment.battery["energy_total_kwh"]
    inv_power_kw = equipment.inverter["rated_power_total_kw"] if equipment.battery["energy_total_kwh"] > 0 else 0.0

    rows: list[dict[str, float | int | str]] = []

    diesel_fuel_l = 0.0
    diesel_energy_kwh = 0.0
    wind_used_total = 0.0
    wind_curtailed_total = 0.0
    battery_charge_total = 0.0
    battery_discharge_total = 0.0
    unserved_total = 0.0
    ballast_total = 0.0
    high_load_total = 0.0

    for i in range(n):
        load_kw = float(load[i])
        mges_kw = float(mges[i])
        wind_av_kw = float(wind_available[i])

        diesel_kw = 0.0
        diesel_units = 0
        diesel_per_unit_kw = 0.0
        fuel_l = 0.0
        batt_charge_kw = 0.0
        batt_discharge_kw = 0.0
        unserved_kw = 0.0
        ballast_kw = 0.0
        curtailed_wind_kw = 0.0

        renewable_kw = mges_kw + wind_av_kw

        if renewable_kw >= load_kw:
            excess_kw = renewable_kw - load_kw
            headroom_kwh = max(0.0, soc_max - soc_energy)
            batt_charge_kw = min(excess_kw, inv_power_kw, headroom_kwh / config.battery_eta_charge if config.battery_eta_charge > 0 else 0.0)
            soc_energy += batt_charge_kw * config.battery_eta_charge

            wind_direct_kw = min(wind_av_kw, max(0.0, load_kw - mges_kw))
            wind_charge_kw = min(max(0.0, wind_av_kw - wind_direct_kw), batt_charge_kw)
            wind_used_kw = wind_direct_kw + wind_charge_kw
            curtailed_wind_kw = max(0.0, wind_av_kw - wind_used_kw)
            ballast_kw = max(0.0, excess_kw - batt_charge_kw)
        else:
            wind_used_kw = wind_av_kw
            deficit_kw = load_kw - renewable_kw

            available_discharge_kw = max(0.0, (soc_energy - soc_min) * config.battery_eta_discharge)
            batt_discharge_kw = min(deficit_kw, inv_power_kw, available_discharge_kw)
            soc_energy -= batt_discharge_kw / config.battery_eta_discharge if config.battery_eta_discharge > 0 else 0.0
            deficit_kw -= batt_discharge_kw

            if deficit_kw > 1e-9:
                des = choose_diesel_dispatch(deficit_kw, config)
                diesel_kw = float(des["total_power_kw"])
                diesel_units = int(des["units"])
                diesel_per_unit_kw = float(des["per_unit_power_kw"])
                fuel_l = float(des["fuel_lph"])
                high_load_total += float(des["high_load_kw"])
                diesel_cover_kw = min(deficit_kw, diesel_kw)
                deficit_kw -= diesel_cover_kw

                surplus_diesel_kw = max(0.0, diesel_kw - diesel_cover_kw)
                extra_charge_kw = 0.0
                if surplus_diesel_kw > 0 and inv_power_kw > 0 and soc_energy < soc_max:
                    headroom_kwh = max(0.0, soc_max - soc_energy)
                    extra_charge_kw = min(surplus_diesel_kw, inv_power_kw - batt_charge_kw, headroom_kwh / config.battery_eta_charge)
                    batt_charge_kw += max(0.0, extra_charge_kw)
                    soc_energy += max(0.0, extra_charge_kw) * config.battery_eta_charge
                ballast_kw += max(0.0, surplus_diesel_kw - max(0.0, extra_charge_kw))

            unserved_kw = max(0.0, deficit_kw)

        soc_energy = min(max(soc_energy, soc_min), soc_max) if soc_max > 0 else 0.0
        soc_fraction = soc_energy / soc_max if soc_max > 0 else 0.0

        diesel_fuel_l += fuel_l
        diesel_energy_kwh += diesel_kw
        wind_used_total += wind_used_kw
        wind_curtailed_total += curtailed_wind_kw
        battery_charge_total += batt_charge_kw
        battery_discharge_total += batt_discharge_kw
        unserved_total += unserved_kw
        ballast_total += ballast_kw

        if make_hourly:
            rows.append({
                "datetime": datetimes[i],
                "load_kwh": load_kw,
                "mges1_kwh": float(mges1[i]),
                "mges2_kwh": float(mges2[i]),
                "mges_total_kwh": mges_kw,
                "wind_speed_10m_mps": float(v10[i]),
                "wind_speed_hub_mps": float(v_hub[i]),
                "wind_available_kwh": wind_av_kw,
                "wind_used_kwh": wind_used_kw,
                "wind_curtailed_kwh": curtailed_wind_kw,
                "battery_charge_kwh": batt_charge_kw,
                "battery_discharge_kwh": batt_discharge_kw,
                "battery_soc_kwh": soc_energy,
                "battery_soc_fraction": soc_fraction,
                "diesel_power_kwh": diesel_kw,
                "diesel_units": diesel_units,
                "diesel_per_unit_kw": diesel_per_unit_kw,
                "diesel_fuel_l": fuel_l,
                "unserved_kwh": unserved_kw,
                "ballast_kwh": ballast_kw,
            })

    hourly = pd.DataFrame(rows) if make_hourly else pd.DataFrame()
    wind_available_total = float(np.sum(wind_available))
    metrics = {
        "load_kwh": float(np.sum(load)),
        "mges_kwh": float(np.sum(mges)),
        "wind_available_kwh": wind_available_total,
        "wind_used_kwh": float(wind_used_total),
        "wind_curtailed_kwh": float(wind_curtailed_total),
        "wind_utilization_fraction": float(wind_used_total / wind_available_total) if wind_available_total > 1e-9 else 0.0,
        "battery_charge_kwh": float(battery_charge_total),
        "battery_discharge_kwh": float(battery_discharge_total),
        "diesel_energy_kwh": float(diesel_energy_kwh),
        "diesel_fuel_l": float(diesel_fuel_l),
        "unserved_kwh": float(unserved_total),
        "ballast_kwh": float(ballast_total),
        "diesel_high_load_kwh": float(high_load_total),
        "equipment_cost_rub": float(equipment.total_price_rub),
        "valid_equipment": 1.0 if equipment.valid else 0.0,
    }
    return SimulationResult(choice=choice, equipment=equipment, hourly=hourly, metrics=metrics)


def monthly_summary(result: SimulationResult) -> pd.DataFrame:
    df = result.hourly.copy()
    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    cols = [
        "load_kwh",
        "mges_total_kwh",
        "wind_used_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "diesel_power_kwh",
        "diesel_fuel_l",
        "unserved_kwh",
        "ballast_kwh",
    ]
    return df.groupby("month", as_index=False)[cols].sum()
