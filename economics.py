from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ALGORITHM_VARIANTS = {
    "Greedy": "Greedy",
    "GWO": "GWO",
    "DE": "DE",
}
VARIANT_COLUMNS = ["Существующая система", "Greedy", "GWO", "DE"]


@dataclass
class EconomySettings:
    mrot_rub_month: float = 27093.0
    district_coefficient: float = 2.0
    staff_count: int = 2
    wind_service_life_years: float = 20.0
    battery_service_life_years: float = 15.0
    inverter_service_life_years: float = 15.0
    project_work_mrot_count: float = 5.0
    wind_repair_percent: float = 3.83
    battery_repair_percent: float = 6.95
    inverter_repair_percent: float = 4.59


@dataclass
class EconomyVariant:
    name: str
    algorithm: str | None
    diesel_fuel_l: float
    fuel_price_rub_l: float
    equipment_cost_rub: float
    wind_cost_rub: float
    battery_cost_rub: float
    inverter_cost_rub: float
    project_work_cost_rub: float
    capital_cost_rub: float
    fuel_cost_rub: float
    amortization_rub_year: float
    repair_cost_rub_year: float
    payroll_rub_year: float
    operating_cost_rub_year: float
    total_first_year_cost_rub: float
    operational_after_purchase_rub_year: float
    payback_years: float | None
    fuel_saving_l: float
    fuel_saving_percent: float
    fuel_saving_rub: float
    wind_info: str
    battery_info: str
    inverter_info: str


@dataclass
class EconomyCalculation:
    variants: dict[str, EconomyVariant]
    input_table: pd.DataFrame
    indicators_table: pd.DataFrame
    savings_table: pd.DataFrame
    summary_text: str


class EconomicsPanel(QWidget):

    def __init__(self) -> None:
        super().__init__()
        self.algorithm_results: Mapping[str, Any] = {}
        self.fuel_price_rub_l: float = 70.0

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        controls = QGroupBox("Параметры экономического расчета")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(8)

        self.mrot_spin = _make_double_spin(0, 1_000_000, 27093.0, 1.0, 0, width=86)
        self.district_coeff_spin = _make_double_spin(0, 10, 2.0, 0.1, 2, width=56)
        self.staff_spin = QSpinBox()
        self.staff_spin.setRange(0, 100)
        self.staff_spin.setValue(2)
        self.staff_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.staff_spin.setFixedWidth(46)
        self.project_work_factor_spin = _make_double_spin(0, 100, 5.0, 0.5, 1, width=52)

        self.wind_life_edit = _make_number_edit(20.0, 1.0, 100.0, 1, width=46)
        self.battery_life_edit = _make_number_edit(15.0, 1.0, 100.0, 1, width=46)
        self.inverter_life_edit = _make_number_edit(15.0, 1.0, 100.0, 1, width=46)
        self.wind_repair_edit = _make_number_edit(3.83, 0.0, 100.0, 2, width=52)
        self.battery_repair_edit = _make_number_edit(6.95, 0.0, 100.0, 2, width=52)
        self.inverter_repair_edit = _make_number_edit(4.59, 0.0, 100.0, 2, width=52)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(_make_labeled_control("МРОТ, руб/мес:", self.mrot_spin))
        top_row.addWidget(_make_labeled_step_control("Районный коэффициент:", self.district_coeff_spin, 0.1))
        top_row.addWidget(_make_labeled_step_control("Персонал, чел:", self.staff_spin, 1))
        top_row.addWidget(_make_labeled_step_control("Проектные работы, МРОТ×коэф:", self.project_work_factor_spin, 0.5))
        top_row.addStretch(1)
        controls_layout.addLayout(top_row)

        life_group = QGroupBox("Срок службы оборудования, лет")
        life_row = QHBoxLayout(life_group)
        life_row.setContentsMargins(8, 6, 8, 6)
        life_row.setSpacing(8)
        life_row.addWidget(_make_labeled_control("ВЭУ:", self.wind_life_edit))
        life_row.addWidget(_make_labeled_control("АКБ:", self.battery_life_edit))
        life_row.addWidget(_make_labeled_control("Инверторы:", self.inverter_life_edit))
        life_row.addStretch(1)

        repair_group = QGroupBox("Плановый ремонт оборудования, %")
        repair_row = QHBoxLayout(repair_group)
        repair_row.setContentsMargins(8, 6, 8, 6)
        repair_row.setSpacing(8)
        repair_row.addWidget(_make_labeled_control("ВЭУ:", self.wind_repair_edit))
        repair_row.addWidget(_make_labeled_control("АКБ:", self.battery_repair_edit))
        repair_row.addWidget(_make_labeled_control("Инверторы:", self.inverter_repair_edit))
        repair_row.addStretch(1)

        equipment_params_row = QHBoxLayout()
        equipment_params_row.setSpacing(10)
        equipment_params_row.addWidget(life_group, 1)
        equipment_params_row.addWidget(repair_group, 1)
        controls_layout.addLayout(equipment_params_row)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        self.refresh_btn = QPushButton("Рассчитать экономику")
        self.refresh_btn.clicked.connect(self.refresh)
        bottom_row.addWidget(self.refresh_btn)

        self.fuel_price_label = QLabel("Цена дизельного топлива берется со вкладки «Расчет».")
        self.fuel_price_label.setWordWrap(True)
        bottom_row.addWidget(self.fuel_price_label, 1)
        controls_layout.addLayout(bottom_row)

        outer.addWidget(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(10)
        self.content_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.content)

        self.message_label = QLabel(
            "После запуска оптимизации нажмите «Рассчитать экономику». "
            "Для полного сравнения выполните расчет Greedy, GWO и DE или кнопку сравнения алгоритмов."
        )
        self.message_label.setWordWrap(True)
        self.content_layout.addWidget(self.message_label)

        self.input_group = QGroupBox("Исходные данные для расчета экономических показателей")
        self.input_layout = QVBoxLayout(self.input_group)
        self.input_layout.setContentsMargins(8, 8, 8, 8)
        self.input_table = QTableWidget()
        _prepare_table_widget(self.input_table)
        self.input_layout.addWidget(self.input_table)
        self.content_layout.addWidget(self.input_group)

        self.indicators_group = QGroupBox("Расчетные экономические показатели")
        self.indicators_layout = QVBoxLayout(self.indicators_group)
        self.indicators_layout.setContentsMargins(8, 8, 8, 8)
        self.indicators_table = QTableWidget()
        _prepare_table_widget(self.indicators_table)
        self.indicators_layout.addWidget(self.indicators_table)
        self.content_layout.addWidget(self.indicators_group)

        self.savings_group = QGroupBox("Топливная экономия и срок окупаемости")
        self.savings_layout = QVBoxLayout(self.savings_group)
        self.savings_layout.setContentsMargins(8, 8, 8, 8)
        self.savings_table = QTableWidget()
        _prepare_table_widget(self.savings_table)
        self.savings_layout.addWidget(self.savings_table)
        self.content_layout.addWidget(self.savings_group)

        self._clear_table_widgets()

    def settings(self) -> EconomySettings:
        return EconomySettings(
            mrot_rub_month=float(self.mrot_spin.value()),
            district_coefficient=float(self.district_coeff_spin.value()),
            staff_count=int(self.staff_spin.value()),
            wind_service_life_years=_read_number_edit(self.wind_life_edit, 20.0),
            battery_service_life_years=_read_number_edit(self.battery_life_edit, 15.0),
            inverter_service_life_years=_read_number_edit(self.inverter_life_edit, 15.0),
            project_work_mrot_count=float(self.project_work_factor_spin.value()),
            wind_repair_percent=_read_number_edit(self.wind_repair_edit, 3.83),
            battery_repair_percent=_read_number_edit(self.battery_repair_edit, 6.95),
            inverter_repair_percent=_read_number_edit(self.inverter_repair_edit, 4.59),
        )

    def set_results(self, algorithm_results: Mapping[str, Any], fuel_price_rub_l: float) -> None:
        self.algorithm_results = dict(algorithm_results)
        self.fuel_price_rub_l = float(fuel_price_rub_l)
        self.fuel_price_label.setText(
            f"Цена дизельного топлива из вкладки «Расчет»: {_fmt_number(self.fuel_price_rub_l, 2)} руб/л."
        )
        self.refresh()

    def clear_results(self) -> None:
        self.algorithm_results = {}
        self._clear_table_widgets()
        self.message_label.setText(
            "После запуска оптимизации нажмите «Рассчитать экономику». "
            "Для полного сравнения выполните расчет Greedy, GWO и DE или кнопку сравнения алгоритмов."
        )

    def refresh(self) -> None:
        if not self.algorithm_results:
            self._clear_table_widgets()
            self.message_label.setText("Нет результатов оптимизации. Сначала выполните расчет хотя бы одного алгоритма.")
            return

        calc = build_economy_calculation(self.algorithm_results, self.settings(), self.fuel_price_rub_l)
        _fill_table_widget(self.input_table, calc.input_table)
        _fill_table_widget(self.indicators_table, calc.indicators_table)
        _fill_table_widget(self.savings_table, calc.savings_table)
        self._set_result_groups_visible(True)
        self.message_label.setText(
            "Экономический расчет выполнен по результатам уже рассчитанных алгоритмов. "
            "Пустые столбцы означают, что соответствующий алгоритм еще не запускался."
        )

    def _clear_table_widgets(self) -> None:
        for table in (self.input_table, self.indicators_table, self.savings_table):
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
            table.setFixedHeight(1)
        self._set_result_groups_visible(False)

    def _set_result_groups_visible(self, visible: bool) -> None:
        for group in (self.input_group, self.indicators_group, self.savings_group):
            group.setVisible(visible)


def build_economy_calculation(
    algorithm_results: Mapping[str, Any],
    settings: EconomySettings,
    fuel_price_rub_l: float,
) -> EconomyCalculation:
    baseline = _first_baseline(algorithm_results)
    base_fuel_l = float(baseline.metrics.get("diesel_fuel_l", 0.0)) if baseline is not None else 0.0
    base_fuel_cost = base_fuel_l * fuel_price_rub_l

    variants: dict[str, EconomyVariant] = {}
    variants["Существующая система"] = EconomyVariant(
        name="Существующая система",
        algorithm=None,
        diesel_fuel_l=base_fuel_l,
        fuel_price_rub_l=fuel_price_rub_l,
        equipment_cost_rub=0.0,
        wind_cost_rub=0.0,
        battery_cost_rub=0.0,
        inverter_cost_rub=0.0,
        project_work_cost_rub=0.0,
        capital_cost_rub=0.0,
        fuel_cost_rub=base_fuel_cost,
        amortization_rub_year=0.0,
        repair_cost_rub_year=0.0,
        payroll_rub_year=0.0,
        operating_cost_rub_year=0.0,
        total_first_year_cost_rub=base_fuel_cost,
        operational_after_purchase_rub_year=base_fuel_cost,
        payback_years=None,
        fuel_saving_l=0.0,
        fuel_saving_percent=0.0,
        fuel_saving_rub=0.0,
        wind_info="нет",
        battery_info="нет",
        inverter_info="нет",
    )

    for algorithm, variant_name in ALGORITHM_VARIANTS.items():
        run_data = algorithm_results.get(algorithm)
        if run_data is None:
            continue

        result = run_data.result
        best = result.best
        e = best.equipment

        diesel_fuel_l = float(best.metrics.get("diesel_fuel_l", 0.0))
        fuel_cost = diesel_fuel_l * fuel_price_rub_l

        wind_cost = _equipment_part_cost(e, "wind")
        battery_cost = _equipment_part_cost(e, "battery")
        inverter_cost = _equipment_part_cost(e, "inverter")
        equipment_cost = wind_cost + battery_cost + inverter_cost
        if equipment_cost <= 1e-9:
            equipment_cost = float(getattr(e, "total_price_rub", 0.0))

        project_cost = settings.mrot_rub_month * settings.district_coefficient * settings.project_work_mrot_count
        capital_cost = equipment_cost + project_cost

        amortization = (
            _safe_div(wind_cost, settings.wind_service_life_years)
            + _safe_div(battery_cost, settings.battery_service_life_years)
            + _safe_div(inverter_cost, settings.inverter_service_life_years)
        )
        repair = (
            wind_cost * settings.wind_repair_percent / 100.0
            + battery_cost * settings.battery_repair_percent / 100.0
            + inverter_cost * settings.inverter_repair_percent / 100.0
        )
        payroll = settings.staff_count * settings.mrot_rub_month * settings.district_coefficient * 12.0
        operating = payroll + repair
        coeff_eff = 0.15
        total_first_year = capital_cost * coeff_eff + fuel_cost + operating
        operational_after_purchase = fuel_cost + operating
        fuel_saving_l = base_fuel_l - diesel_fuel_l
        fuel_saving_percent = (fuel_saving_l / base_fuel_l * 100.0) if base_fuel_l > 1e-9 else 0.0
        fuel_saving_rub = base_fuel_cost - fuel_cost
        payback = capital_cost / fuel_saving_rub if fuel_saving_rub > 1e-9 else None

        variants[variant_name] = EconomyVariant(
            name=variant_name,
            algorithm=algorithm,
            diesel_fuel_l=diesel_fuel_l,
            fuel_price_rub_l=fuel_price_rub_l,
            equipment_cost_rub=equipment_cost,
            wind_cost_rub=wind_cost,
            battery_cost_rub=battery_cost,
            inverter_cost_rub=inverter_cost,
            project_work_cost_rub=project_cost,
            capital_cost_rub=capital_cost,
            fuel_cost_rub=fuel_cost,
            amortization_rub_year=amortization,
            repair_cost_rub_year=repair,
            payroll_rub_year=payroll,
            operating_cost_rub_year=operating,
            total_first_year_cost_rub=total_first_year,
            operational_after_purchase_rub_year=operational_after_purchase,
            payback_years=payback,
            fuel_saving_l=fuel_saving_l,
            fuel_saving_percent=fuel_saving_percent,
            fuel_saving_rub=fuel_saving_rub,
            wind_info=_wind_info(e),
            battery_info=_battery_info(e),
            inverter_info=_inverter_info(e),
        )

    input_table = _build_input_table(variants, fuel_price_rub_l)
    indicators_table = _build_indicators_table(variants)
    savings_table = _build_savings_table(variants)
    summary_text = _build_summary_text(variants)
    return EconomyCalculation(variants, input_table, indicators_table, savings_table, summary_text)


def _first_baseline(algorithm_results: Mapping[str, Any]) -> Any | None:
    for data in algorithm_results.values():
        baseline = getattr(data, "baseline", None)
        if baseline is not None:
            return baseline
    return None


def _build_input_table(variants: dict[str, EconomyVariant], fuel_price: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("Расход топлива ДЭС, л", variants, lambda v: v.diesel_fuel_l),
            {"Показатель": "Цена дизельного топлива, руб/л", **{col: _fmt_number(fuel_price, 2) for col in VARIANT_COLUMNS}},
            _row("ВЭУ, шт. и модель", variants, lambda v: v.wind_info),
            _row("АКБ, шт. и модель", variants, lambda v: v.battery_info),
            _row("Инверторы, шт. и модель", variants, lambda v: v.inverter_info),
        ]
    )


def _build_indicators_table(variants: dict[str, EconomyVariant]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("Капитальные затраты на ВЭУ, руб/год", variants, lambda v: v.wind_cost_rub),
            _row("Капитальные затраты на АКБ, руб/год", variants, lambda v: v.battery_cost_rub),
            _row("Капитальные затраты на инверторы, руб/год", variants, lambda v: v.inverter_cost_rub),
            _row("Стоимость оборудования Kоб, руб/год", variants, lambda v: v.equipment_cost_rub),
            _row("Проектные работы Cпр, руб/год", variants, lambda v: v.project_work_cost_rub),
            _row("Капитальные затраты K, руб/год", variants, lambda v: v.capital_cost_rub),
            _row("Топливные издержки Cт, руб/год", variants, lambda v: v.fuel_cost_rub),
            _row("Амортизация A, руб/год", variants, lambda v: v.amortization_rub_year),
            _row("Плановый ремонт, руб/год", variants, lambda v: v.repair_cost_rub_year),
            _row("ФОТ, руб/год", variants, lambda v: v.payroll_rub_year),
            _row("Эксплуатационные издержки Иэкспл, руб/год", variants, lambda v: v.operating_cost_rub_year),
            _row("Итоговые суммарные затраты первого года, руб/год", variants, lambda v: v.total_first_year_cost_rub),
        ]
    )


def _build_savings_table(variants: dict[str, EconomyVariant]) -> pd.DataFrame:
    project_cols = ["Greedy", "GWO", "DE"]

    def row(label: str, getter):
        out = {"Показатель": label}
        for col in project_cols:
            variant = variants.get(col)
            out[col] = getter(variant) if variant else "нет данных"
        return out

    return pd.DataFrame(
        [
            row("Снижение расхода топлива, л", lambda v: v.fuel_saving_l),
            row("Снижение расхода топлива, %", lambda v: v.fuel_saving_percent),
            row("Экономия только по топливу, руб/год", lambda v: v.fuel_saving_rub),
            row("Операционные затраты после года закупки, руб/год", lambda v: v.operational_after_purchase_rub_year),
            row("Простой срок окупаемости, лет", lambda v: v.payback_years if v.payback_years is not None else "не окупается"),
        ]
    )


def _row(label: str, variants: dict[str, EconomyVariant], getter) -> dict[str, object]:
    out: dict[str, object] = {"Показатель": label}
    for col in VARIANT_COLUMNS:
        variant = variants.get(col)
        out[col] = getter(variant) if variant else "нет данных"
    return out


def _build_summary_text(variants: dict[str, EconomyVariant]) -> str:
    existing = variants.get("Существующая система")
    project_variants = [v for key, v in variants.items() if key != "Существующая система"]
    if not project_variants:
        return "Для экономического сравнения пока нет рассчитанных алгоритмов."

    best = min(project_variants, key=lambda v: v.total_first_year_cost_rub)
    max_fuel_saving = max(project_variants, key=lambda v: v.fuel_saving_l)
    lines = [
        f"Минимальные итоговые суммарные затраты первого года среди рассчитанных проектируемых схем получены у {best.name} ({best.algorithm}): {_fmt_number(best.total_first_year_cost_rub, 1)} руб/год",
        f"Максимальное снижение расхода топлива среди рассчитанных схем получено у {max_fuel_saving.name} ({max_fuel_saving.algorithm}): {_fmt_number(max_fuel_saving.fuel_saving_l, 1)} л, или {_fmt_number(max_fuel_saving.fuel_saving_percent, 2)} %.",
    ]
    if existing is not None:
        diff = best.total_first_year_cost_rub - existing.total_first_year_cost_rub
        if diff > 0:
            lines.append(
                f"Относительно существующей схемы первый год внедрения для лучшего рассчитанного алгоритма дороже на {_fmt_number(diff, 1)} руб/год, так как капитальные затраты учитываются полной суммой."
            )
        else:
            lines.append(
                f"Относительно существующей схемы лучший рассчитанный алгоритм дешевле уже в первый год на {_fmt_number(abs(diff), 1)} руб/год"
            )
    if best.payback_years is not None:
        lines.append(f"Простой срок окупаемости лучшего рассчитанного алгоритма: {_fmt_number(best.payback_years, 2)} года.")
    else:
        lines.append("Для лучшего рассчитанного алгоритма простой срок окупаемости не определяется, так как экономия топлива отсутствует или отрицательна.")
    return "\n".join(lines)


def _wind_info(e: Any) -> str:
    model = e.wind.get("model", "-")
    count = int(round(float(e.wind.get("count", 0))))
    return f"{model}, {count} шт."


def _battery_info(e: Any) -> str:
    model = e.battery.get("model", "-")
    count = int(round(float(e.battery.get("total_cells", 0))))
    return f"{model}, {count} шт."


def _inverter_info(e: Any) -> str:
    model = e.inverter.get("model", "-")
    count = int(round(float(e.inverter.get("count", 0))))
    return f"{model}, {count} шт."


def _equipment_part_cost(e: Any, part_name: str) -> float:
    part = getattr(e, part_name, {})
    if isinstance(part, dict):
        return float(part.get("price_total_rub", 0.0))
    return 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def _make_number_edit(value: float, minimum: float, maximum: float, decimals: int, width: int = 52) -> QLineEdit:
    edit = QLineEdit()
    edit.setText(_fmt_plain_number(value, decimals))
    validator = QDoubleValidator(minimum, maximum, decimals)
    validator.setNotation(QDoubleValidator.StandardNotation)
    edit.setValidator(validator)
    edit.setFixedWidth(width)
    return edit


def _read_number_edit(edit: QLineEdit, default: float) -> float:
    text = edit.text().strip().replace(" ", "").replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        value = float(default)
        edit.setText(_fmt_plain_number(value, 2))
    return value


def _fmt_plain_number(value: float, decimals: int) -> str:
    text = f"{float(value):.{decimals}f}"
    if decimals > 0:
        text = text.rstrip("0").rstrip(".")
    return text


def _make_double_spin(minimum: float, maximum: float, value: float, step: float, decimals: int, width: int = 78) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
    spin.setFixedWidth(width)
    return spin


def _make_labeled_control(label_text: str, control: QWidget) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(control)
    return widget


def _make_labeled_step_control(label_text: str, spin: QDoubleSpinBox | QSpinBox, step: float | int) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(_make_step_control(spin, step))
    return widget


def _make_step_control(spin: QDoubleSpinBox | QSpinBox, step: float | int) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)

    minus_btn = QPushButton("−")
    plus_btn = QPushButton("+")
    minus_btn.setFixedWidth(26)
    plus_btn.setFixedWidth(26)
    minus_btn.setFixedHeight(24)
    plus_btn.setFixedHeight(24)
    minus_btn.clicked.connect(lambda: spin.setValue(spin.value() - step))
    plus_btn.clicked.connect(lambda: spin.setValue(spin.value() + step))

    layout.addWidget(spin)
    layout.addWidget(minus_btn)
    layout.addWidget(plus_btn)
    return widget


def _prepare_table_widget(table: QTableWidget) -> None:
    table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    table.setWordWrap(True)
    table.setShowGrid(True)
    table.setMinimumHeight(1)
    table.setFixedHeight(1)


def _fill_table_widget(table: QTableWidget, df: pd.DataFrame) -> None:
    table.clear()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(col) for col in df.columns])
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectItems)
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for col_idx, col_name in enumerate(df.columns):
            value = row[col_name]
            item = QTableWidgetItem(_format_cell_value(value, is_label_col=(col_idx == 0)))
            item.setTextAlignment(Qt.AlignCenter)
            if col_idx == 0:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(row_idx, col_idx, item)

    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Stretch)
    header.setStretchLastSection(False)
    table.verticalHeader().setVisible(False)
    table.resizeRowsToContents()
    _resize_table_to_contents(table)


def _resize_table_to_contents(table: QTableWidget) -> None:
    header_height = table.horizontalHeader().height() if not table.horizontalHeader().isHidden() else 0
    rows_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
    frame = 2 * table.frameWidth()
    height = header_height + rows_height + frame + 4
    table.setFixedHeight(max(1, height))


def _format_cell_value(value: object, is_label_col: bool = False) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not isfinite(float(value)):
            return "—"
        decimals = 2 if abs(float(value)) < 100 and not float(value).is_integer() else 1
        return _fmt_number(float(value), decimals)
    return str(value)


def _fmt_number(value: float, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}".replace(",", " ").replace(".", ",")
