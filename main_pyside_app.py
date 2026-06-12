from __future__ import annotations

import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from data_loader import DataLoadError, SourceData, load_source_data
from economics import EconomicsPanel
from energy_models import EquipmentChoice, SimulationResult, simulate_system
from optimizer import OptimizationConfig, OptimizationResult, run_optimization
from plotting import GraphsPanel, build_existing_year_figure, build_final_day_figure, build_final_year_figure
from tables import TablesPanel, build_existing_year_table, build_final_day_table, build_final_year_table

AlgorithmName = Literal["Greedy", "GWO", "DE"]
ALGORITHMS: list[AlgorithmName] = ["Greedy", "GWO", "DE"]

GRAPH_CHOICES = [
    ("existing_year", "Годовой исходный"),
    ("final_year", "Годовой итоговый"),
    ("jan15", "15 января"),
    ("apr15", "15 апреля"),
    ("jul15", "15 июля"),
    ("oct15", "15 октября"),
]

TABLE_CHOICES = [
    ("existing_year", "Годовая исходная"),
    ("final_year", "Годовая итоговая"),
    ("jan15", "15 января"),
    ("apr15", "15 апреля"),
    ("jul15", "15 июля"),
    ("oct15", "15 октября"),
]

TARGET_CHOICES = ["Последний рассчитанный", "Greedy", "GWO", "DE", "Все алгоритмы"]

APP_ID = "hybrid_ases.optimizer"


def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent / relative_path


def get_app_icon() -> QIcon:
    for icon_name in ("icon.ico", "app_icon.png"):
        icon_path = resource_path(icon_name)
        if icon_path.exists():
            return QIcon(str(icon_path))
    return QIcon()


def apply_windows_taskbar_icon() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


@dataclass
class RunData:
    result: OptimizationResult
    baseline: SimulationResult


class OptimizationWorker(QThread):

    finished_ok = Signal(str, str, object, object, object)
    failed = Signal(str)

    def __init__(self, excel_path: str, algorithm: str, fuel_price_rub_l: float):
        super().__init__()
        self.excel_path = excel_path
        self.algorithm = algorithm
        self.fuel_price_rub_l = float(fuel_price_rub_l)

    def run(self) -> None:
        try:
            data, baseline, result = _run_single_calculation(self.excel_path, self.algorithm)
            _save_result_files(self.excel_path, self.algorithm, result)
            text = format_optimization_report(result, baseline, self.fuel_price_rub_l)
            self.finished_ok.emit(self.algorithm, text, data, result, baseline)
        except Exception as exc:
            self.failed.emit(str(exc))


class CompareWorker(QThread):

    finished_ok = Signal(object, object, object, object)
    failed = Signal(str)

    def __init__(self, excel_path: str, fuel_price_rub_l: float):
        super().__init__()
        self.excel_path = excel_path
        self.fuel_price_rub_l = float(fuel_price_rub_l)

    def run(self) -> None:
        try:
            data = load_source_data(self.excel_path)
            config = OptimizationConfig()
            baseline = _simulate_baseline(data, config)
            texts: dict[str, str] = {}
            results: dict[str, OptimizationResult] = {}
            for algorithm in ALGORITHMS:
                result = run_optimization(data, algorithm=algorithm, config=OptimizationConfig())
                _save_result_files(self.excel_path, algorithm, result)
                texts[algorithm] = format_optimization_report(result, baseline, self.fuel_price_rub_l)
                results[algorithm] = result
            self.finished_ok.emit(texts, data, results, baseline)
        except Exception as exc:
            self.failed.emit(str(exc))


class CustomEquipmentOptimizationWorker(QThread):

    finished_ok = Signal(str, str, object, object, object)
    failed = Signal(str)

    def __init__(self, data: SourceData, results_dir: str | Path, algorithm: str, fuel_price_rub_l: float):
        super().__init__()
        self.data = data
        self.results_dir = Path(results_dir)
        self.algorithm = algorithm
        self.fuel_price_rub_l = float(fuel_price_rub_l)

    def run(self) -> None:
        try:
            data, baseline, result = _run_custom_calculation(self.data, self.algorithm)
            _save_result_files_to_dir(self.results_dir, self.algorithm, result)
            text = format_optimization_report(result, baseline, self.fuel_price_rub_l)
            self.finished_ok.emit(self.algorithm, text, data, result, baseline)
        except Exception as exc:
            self.failed.emit(str(exc))


class CustomEquipmentCompareWorker(QThread):

    finished_ok = Signal(object, object, object, object)
    failed = Signal(str)

    def __init__(self, data: SourceData, results_dir: str | Path, fuel_price_rub_l: float):
        super().__init__()
        self.data = data
        self.results_dir = Path(results_dir)
        self.fuel_price_rub_l = float(fuel_price_rub_l)

    def run(self) -> None:
        try:
            data = self.data
            config = OptimizationConfig()
            baseline = _simulate_baseline(data, config)
            texts: dict[str, str] = {}
            results: dict[str, OptimizationResult] = {}
            for algorithm in ALGORITHMS:
                result = run_optimization(data, algorithm=algorithm, config=OptimizationConfig())
                _save_result_files_to_dir(self.results_dir, algorithm, result)
                texts[algorithm] = format_optimization_report(result, baseline, self.fuel_price_rub_l)
                results[algorithm] = result
            self.finished_ok.emit(texts, data, results, baseline)
        except Exception as exc:
            self.failed.emit(str(exc))


def empty_wind_turbines_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "model",
            "rated_power_kw",
            "hub_height_m",
            "price_rub",
        ]
    )


def empty_batteries_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "model",
            "voltage_v",
            "capacity_ah",
            "energy_kwh",
            "specific_energy_wh_kg",
            "specific_energy_wh_l",
            "mass_kg",
            "price_rub",
        ]
    )


def empty_inverters_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "model",
            "rated_power_kw",
            "max_power_kw",
            "peak_power_kw",
            "input_voltage_v",
            "output_voltage_v",
            "max_capacity_ah",
            "price_rub",
        ]
    )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Оптимизация гибридной АСЭС")
        self.setWindowIcon(get_app_icon())
        self.resize(1260, 860)

        self.worker: OptimizationWorker | CompareWorker | None = None
        self.current_data: SourceData | None = None
        self.baseline: SimulationResult | None = None
        self.results_by_algorithm: dict[str, OptimizationResult] = {}
        self.run_data_by_algorithm: dict[str, RunData] = {}
        self.last_algorithm: str | None = None
        self.manual_wind_turbines = empty_wind_turbines_catalog()
        self.manual_batteries = empty_batteries_catalog()
        self.manual_inverters = empty_inverters_catalog()
        self.equipment_catalogs_cleared = False

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.main_tabs = QTabWidget()
        root_layout.addWidget(self.main_tabs)

        self.control_tab = QWidget()
        self.graphs_tab = GraphsPanel()
        self.tables_tab = TablesPanel()
        self.economics_tab = EconomicsPanel()

        self.main_tabs.addTab(self.control_tab, "Расчет")
        self.main_tabs.addTab(self.graphs_tab, "Графики")
        self.main_tabs.addTab(self.tables_tab, "Таблицы")
        self.main_tabs.addTab(self.economics_tab, "Экономика")

        self._build_control_tab()
        self._apply_light_style()

    # -------------------------- интерфейс --------------------------

    def _build_control_tab(self) -> None:
        layout = QVBoxLayout(self.control_tab)
        layout.setSpacing(8)

        file_group = QGroupBox("Исходные данные")
        file_layout = QGridLayout(file_group)
        self.excel_path_edit = QLineEdit()
        self.excel_path_edit.setPlaceholderText("Выберите файл с исходными данными")
        browse_btn = QPushButton("Выбрать Excel")
        browse_btn.clicked.connect(self.choose_excel)
        file_layout.addWidget(QLabel("Файл Excel:"), 0, 0)
        file_layout.addWidget(self.excel_path_edit, 0, 1)
        file_layout.addWidget(browse_btn, 0, 2)
        layout.addWidget(file_group)

        top_row = QHBoxLayout()

        opt_group = QGroupBox("Параметры оптимизации")
        opt_layout = QGridLayout(opt_group)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(ALGORITHMS)
        opt_layout.addWidget(QLabel("Алгоритм:"), 0, 0)
        opt_layout.addWidget(self.algorithm_combo, 0, 1)

        self.fuel_price_spin = QDoubleSpinBox()
        self.fuel_price_spin.setRange(0.0, 10000.0)
        self.fuel_price_spin.setDecimals(2)
        self.fuel_price_spin.setSingleStep(0.01)
        self.fuel_price_spin.setValue(96.51)
        self.fuel_price_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fuel_price_spin.setMinimumWidth(110)
        fuel_control = self._make_step_control(self.fuel_price_spin, 0.01)
        opt_layout.addWidget(QLabel("Цена дизельного топлива, руб/л:"), 1, 0)
        opt_layout.addWidget(fuel_control, 1, 1)
        top_row.addWidget(opt_group, 1)

        view_group = QGroupBox("Графики и таблицы")
        view_layout = QGridLayout(view_group)
        self.target_algorithm_combo = QComboBox()
        self.target_algorithm_combo.addItems(TARGET_CHOICES)
        self.graph_combo = QComboBox()
        for key, title in GRAPH_CHOICES:
            self.graph_combo.addItem(title, key)
        self.table_combo = QComboBox()
        for key, title in TABLE_CHOICES:
            self.table_combo.addItem(title, key)

        self.open_graph_btn = QPushButton("Открыть выбранный график")
        self.open_graph_btn.clicked.connect(self.open_selected_graph)
        self.open_all_graphs_btn = QPushButton("Открыть все графики")
        self.open_all_graphs_btn.clicked.connect(self.open_all_graphs)
        self.open_table_btn = QPushButton("Открыть выбранную таблицу")
        self.open_table_btn.clicked.connect(self.open_selected_table)
        self.open_all_tables_btn = QPushButton("Открыть все таблицы")
        self.open_all_tables_btn.clicked.connect(self.open_all_tables)

        view_layout.addWidget(QLabel("Алгоритм вывода:"), 0, 0)
        view_layout.addWidget(self.target_algorithm_combo, 0, 1, 1, 3)
        view_layout.addWidget(QLabel("График:"), 1, 0)
        view_layout.addWidget(self.graph_combo, 1, 1)
        view_layout.addWidget(self.open_graph_btn, 1, 2)
        view_layout.addWidget(self.open_all_graphs_btn, 1, 3)
        view_layout.addWidget(QLabel("Таблица:"), 2, 0)
        view_layout.addWidget(self.table_combo, 2, 1)
        view_layout.addWidget(self.open_table_btn, 2, 2)
        view_layout.addWidget(self.open_all_tables_btn, 2, 3)
        top_row.addWidget(view_group, 1)
        layout.addLayout(top_row)

        data_buttons = QHBoxLayout()
        self.load_btn = QPushButton("Проверить загруженный исходный файл")
        self.load_btn.clicked.connect(self.check_loading)

        self.manual_equipment_btn = QPushButton("Загрузить пользовательское оборудование")
        self.manual_equipment_btn.clicked.connect(self.open_manual_equipment_dialog)

        self.clear_equipment_btn = QPushButton("Очистить только оборудование")
        self.clear_equipment_btn.clicked.connect(self.clear_equipment_only)

        data_buttons.addWidget(self.load_btn)
        data_buttons.addWidget(self.manual_equipment_btn)
        data_buttons.addWidget(self.clear_equipment_btn)
        data_buttons.addStretch(1)
        layout.addLayout(data_buttons)

        outputs = QGridLayout()
        self.load_output = QPlainTextEdit()
        self.load_output.setReadOnly(True)
        self.load_output.setPlaceholderText("Информация о загрузке исходных данных.")
        self.load_output.setMinimumHeight(75)
        self.load_output.setMaximumHeight(95)
        load_output_group = _group_with_text("Информация о загрузке исходных данных", self.load_output)
        load_output_group.setMaximumHeight(150)
        outputs.addWidget(load_output_group, 0, 0, 1, 3)
        outputs.setRowStretch(0, 0)
        outputs.setRowStretch(1, 0)
        outputs.setRowStretch(2, 0)
        outputs.setRowStretch(3, 1)

        optimization_buttons = QHBoxLayout()
        self.run_btn = QPushButton("Запустить оптимизацию")
        self.run_btn.clicked.connect(self.run_optimization)
        self.compare_btn = QPushButton("Сравнить результаты алгоритмов оптимизации")
        self.compare_btn.clicked.connect(self.compare_algorithms)

        optimization_buttons.addWidget(self.run_btn)
        optimization_buttons.addWidget(self.compare_btn)
        optimization_buttons.addStretch(1)
        outputs.addLayout(optimization_buttons, 1, 0, 1, 3)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        outputs.addWidget(self.progress, 2, 0, 1, 3)

        self.algorithm_outputs: dict[str, QPlainTextEdit] = {}
        for col, algorithm in enumerate(ALGORITHMS):
            edit = QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(f"Здесь появится результат оптимизации {algorithm}.")
            self.algorithm_outputs[algorithm] = edit
            outputs.addWidget(_group_with_text(f"Результат оптимизации {algorithm}", edit), 3, col)
        layout.addLayout(outputs, 1)

    def _make_number_spin(self, value: float = 0.0, decimals: int = 1, width: int = 95, maximum: float = 1_000_000_000_000.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(decimals)
        spin.setValue(float(value))
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spin.setFixedWidth(width)
        return spin

    def open_manual_equipment_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Пользовательское оборудование")
        dialog.setFixedSize(690, 340)

        root = QVBoxLayout(dialog)
        root.setSpacing(8)

        note = QLabel(
            "Добавленное оборудование будет использоваться вместе с каталогами из Excel. "
            "Если каталоги оборудования очищены, расчет будет выполняться только по пользовательским позициям."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # ВЭУ
        wt_tab = QWidget()
        wt_grid = QGridLayout(wt_tab)
        wt_grid.setHorizontalSpacing(8)
        wt_grid.setVerticalSpacing(6)
        wt_model = QLineEdit()
        wt_model.setPlaceholderText("Марка ВЭУ")
        wt_power = self._make_number_spin(0.0, 1, 95)
        wt_height = self._make_number_spin(0.0, 1, 95)
        wt_price = self._make_number_spin(0.0, 1, 125)
        wt_add = QPushButton("Добавить ВЭУ")

        wt_grid.addWidget(QLabel("Марка:"), 0, 0)
        wt_grid.addWidget(wt_model, 0, 1, 1, 3)
        wt_grid.addWidget(QLabel("Pном, кВт:"), 1, 0)
        wt_grid.addWidget(wt_power, 1, 1)
        wt_grid.addWidget(QLabel("Высота оси, м:"), 1, 2)
        wt_grid.addWidget(wt_height, 1, 3)
        wt_grid.addWidget(QLabel("Цена, руб:"), 2, 0)
        wt_grid.addWidget(wt_price, 2, 1)
        wt_grid.addWidget(wt_add, 2, 2, 1, 2)
        tabs.addTab(wt_tab, "ВЭУ")

        # АКБ
        bat_tab = QWidget()
        bat_grid = QGridLayout(bat_tab)
        bat_grid.setHorizontalSpacing(8)
        bat_grid.setVerticalSpacing(6)
        bat_model = QLineEdit()
        bat_model.setPlaceholderText("Марка АКБ")
        bat_voltage = self._make_number_spin(0.0, 1, 95)
        bat_capacity = self._make_number_spin(0.0, 1, 95)
        bat_price = self._make_number_spin(0.0, 1, 125)
        bat_add = QPushButton("Добавить АКБ")

        bat_grid.addWidget(QLabel("Марка:"), 0, 0)
        bat_grid.addWidget(bat_model, 0, 1, 1, 3)
        bat_grid.addWidget(QLabel("U, В:"), 1, 0)
        bat_grid.addWidget(bat_voltage, 1, 1)
        bat_grid.addWidget(QLabel("C, А·ч:"), 1, 2)
        bat_grid.addWidget(bat_capacity, 1, 3)
        bat_grid.addWidget(QLabel("Цена, руб:"), 2, 0)
        bat_grid.addWidget(bat_price, 2, 1)
        bat_grid.addWidget(bat_add, 2, 2, 1, 2)
        tabs.addTab(bat_tab, "АКБ")

        # Инверторы
        inv_tab = QWidget()
        inv_grid = QGridLayout(inv_tab)
        inv_grid.setHorizontalSpacing(8)
        inv_grid.setVerticalSpacing(6)
        inv_model = QLineEdit()
        inv_model.setPlaceholderText("Марка инвертора")
        inv_pnom = self._make_number_spin(0.0, 1, 95)
        inv_pmax = self._make_number_spin(0.0, 1, 95)
        inv_ppeak = self._make_number_spin(0.0, 1, 95)
        inv_uin = self._make_number_spin(0.0, 1, 95)
        inv_uout = self._make_number_spin(0.0, 1, 95)
        inv_cmax = self._make_number_spin(0.0, 1, 95)
        inv_price = self._make_number_spin(0.0, 1, 125)
        inv_add = QPushButton("Добавить инвертор")

        inv_grid.addWidget(QLabel("Марка:"), 0, 0)
        inv_grid.addWidget(inv_model, 0, 1, 1, 5)
        inv_grid.addWidget(QLabel("Pном, кВт:"), 1, 0)
        inv_grid.addWidget(inv_pnom, 1, 1)
        inv_grid.addWidget(QLabel("Pmax, кВт:"), 1, 2)
        inv_grid.addWidget(inv_pmax, 1, 3)
        inv_grid.addWidget(QLabel("Pпик, кВт:"), 1, 4)
        inv_grid.addWidget(inv_ppeak, 1, 5)
        inv_grid.addWidget(QLabel("Uвх, В:"), 2, 0)
        inv_grid.addWidget(inv_uin, 2, 1)
        inv_grid.addWidget(QLabel("Uвых, В:"), 2, 2)
        inv_grid.addWidget(inv_uout, 2, 3)
        inv_grid.addWidget(QLabel("Cmax, А·ч:"), 2, 4)
        inv_grid.addWidget(inv_cmax, 2, 5)
        inv_grid.addWidget(QLabel("Цена, руб:"), 3, 0)
        inv_grid.addWidget(inv_price, 3, 1)
        inv_grid.addWidget(inv_add, 3, 2, 1, 4)
        tabs.addTab(inv_tab, "Инверторы")

        def add_wind() -> None:
            model = wt_model.text().strip() or "Пользовательская ВЭУ"
            row = {
                "id": 0,
                "model": model,
                "rated_power_kw": float(wt_power.value()),
                "hub_height_m": float(wt_height.value()),
                "price_rub": float(wt_price.value()),
            }
            self._add_manual_wind_turbine_row(row, "Добавлена ВЭУ: " + model)
            QMessageBox.information(dialog, "Пользовательское оборудование", "ВЭУ добавлена в каталог.")

        def add_battery() -> None:
            model = bat_model.text().strip() or "Пользовательская АКБ"
            voltage = float(bat_voltage.value())
            capacity = float(bat_capacity.value())
            row = {
                "id": 0,
                "model": model,
                "voltage_v": voltage,
                "capacity_ah": capacity,
                "energy_kwh": voltage * capacity / 1000.0,
                "specific_energy_wh_kg": 0.0,
                "specific_energy_wh_l": 0.0,
                "mass_kg": 0.0,
                "price_rub": float(bat_price.value()),
            }
            self._add_manual_battery_row(row, "Добавлена АКБ: " + model)
            QMessageBox.information(dialog, "Пользовательское оборудование", "АКБ добавлена в каталог.")

        def add_inverter() -> None:
            model = inv_model.text().strip() or "Пользовательский инвертор"
            row = {
                "id": 0,
                "model": model,
                "rated_power_kw": float(inv_pnom.value()),
                "max_power_kw": float(inv_pmax.value()),
                "peak_power_kw": float(inv_ppeak.value()),
                "input_voltage_v": float(inv_uin.value()),
                "output_voltage_v": float(inv_uout.value()),
                "max_capacity_ah": float(inv_cmax.value()),
                "price_rub": float(inv_price.value()),
            }
            self._add_manual_inverter_row(row, "Добавлен инвертор: " + model)
            QMessageBox.information(dialog, "Пользовательское оборудование", "Инвертор добавлен в каталог.")

        wt_add.clicked.connect(add_wind)
        bat_add.clicked.connect(add_battery)
        inv_add.clicked.connect(add_inverter)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

        dialog.exec()

    def _make_step_control(self, spin: QDoubleSpinBox, step: float) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        minus_btn = QPushButton("−")
        plus_btn = QPushButton("+")
        minus_btn.setFixedWidth(34)
        plus_btn.setFixedWidth(34)
        minus_btn.clicked.connect(lambda: spin.setValue(spin.value() - step))
        plus_btn.clicked.connect(lambda: spin.setValue(spin.value() + step))
        layout.addWidget(spin)
        layout.addWidget(minus_btn)
        layout.addWidget(plus_btn)
        return widget

    def _add_manual_wind_turbine_row(self, row: dict, message: str) -> None:
        self.manual_wind_turbines = pd.concat([self.manual_wind_turbines, pd.DataFrame([row])], ignore_index=True)
        if self.current_data is not None:
            self.current_data.wind_turbines = _append_equipment_rows(
                self.current_data.wind_turbines,
                pd.DataFrame([row]),
                empty_wind_turbines_catalog,
            )
        self._equipment_changed(message)

    def _add_manual_battery_row(self, row: dict, message: str) -> None:
        self.manual_batteries = pd.concat([self.manual_batteries, pd.DataFrame([row])], ignore_index=True)
        if self.current_data is not None:
            self.current_data.batteries = _append_equipment_rows(
                self.current_data.batteries,
                pd.DataFrame([row]),
                empty_batteries_catalog,
            )
        self._equipment_changed(message)

    def _add_manual_inverter_row(self, row: dict, message: str) -> None:
        self.manual_inverters = pd.concat([self.manual_inverters, pd.DataFrame([row])], ignore_index=True)
        if self.current_data is not None:
            self.current_data.inverters = _append_equipment_rows(
                self.current_data.inverters,
                pd.DataFrame([row]),
                empty_inverters_catalog,
            )
        self._equipment_changed(message)

    def clear_equipment_only(self) -> None:
        self.equipment_catalogs_cleared = True
        self.manual_wind_turbines = empty_wind_turbines_catalog()
        self.manual_batteries = empty_batteries_catalog()
        self.manual_inverters = empty_inverters_catalog()
        if self.current_data is not None:
            self.current_data.wind_turbines = empty_wind_turbines_catalog()
            self.current_data.batteries = empty_batteries_catalog()
            self.current_data.inverters = empty_inverters_catalog()
        self._clear_calculation_results_after_equipment_change()
        self._show_data_status("Каталоги оборудования очищены. Исходные режимные данные сохранены.")

    def _equipment_changed(self, message: str) -> None:
        self._clear_calculation_results_after_equipment_change()
        self._show_data_status(message)

    def _clear_calculation_results_after_equipment_change(self) -> None:
        self.baseline = None
        self.results_by_algorithm.clear()
        self.run_data_by_algorithm.clear()
        self.last_algorithm = None
        self.graphs_tab.clear_figures()
        self.tables_tab.clear_tables()
        self.economics_tab.clear_results()
        for edit in self.algorithm_outputs.values():
            edit.clear()

    def _has_manual_equipment(self) -> bool:
        return (
            not self.manual_wind_turbines.empty
            or not self.manual_batteries.empty
            or not self.manual_inverters.empty
        )

    def _uses_custom_equipment(self) -> bool:
        return self.equipment_catalogs_cleared or self._has_manual_equipment()

    def _load_data_with_current_equipment(self, excel_path: Path) -> SourceData:
        data = load_source_data(excel_path)
        if self.equipment_catalogs_cleared:
            data.wind_turbines = empty_wind_turbines_catalog()
            data.batteries = empty_batteries_catalog()
            data.inverters = empty_inverters_catalog()
        return self._append_manual_equipment_to_data(data)

    def _append_manual_equipment_to_data(self, data: SourceData) -> SourceData:
        data.wind_turbines = _append_equipment_rows(
            data.wind_turbines,
            self.manual_wind_turbines,
            empty_wind_turbines_catalog,
        )
        data.batteries = _append_equipment_rows(
            data.batteries,
            self.manual_batteries,
            empty_batteries_catalog,
        )
        data.inverters = _append_equipment_rows(
            data.inverters,
            self.manual_inverters,
            empty_inverters_catalog,
        )
        return data

    def _show_data_status(self, prefix: str) -> None:
        if self.current_data is not None:
            self.load_output.setPlainText(prefix + "\n\n" + equipment_catalog_summary(self.current_data))
            return
        self.load_output.setPlainText(
            prefix
            + "\n\nИсходный файл с графиками нагрузки, МГЭС, ДЭС и ветра еще не загружен. "
            + "Оборудование сохранено и будет добавлено к данным после загрузки Excel.\n\n"
            + manual_equipment_summary(self.manual_wind_turbines, self.manual_batteries, self.manual_inverters)
        )

    def _prepare_custom_data_for_calculation(self) -> tuple[SourceData, Path]:
        excel_path = Path(self._excel_path())
        results_dir = excel_path.parent
        data = self._load_data_with_current_equipment(excel_path)
        _ensure_equipment_ready(data)
        self.current_data = data
        self._show_data_status("Исходные данные и пользовательские каталоги оборудования готовы к расчету.")
        return data, results_dir

    def _apply_light_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #f6f6f6; color: black; }
            QGroupBox { border: 1px solid #b8b8b8; border-radius: 5px; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget { background-color: white; color: black; border: 1px solid #b8b8b8; }
            QPushButton { background-color: #eeeeee; color: black; border: 1px solid #a9a9a9; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #b8b8b8; background-color: #f6f6f6; }
            QTabBar::tab { background: #eeeeee; color: black; border: 1px solid #b8b8b8; padding: 6px 14px; }
            QTabBar::tab:selected { background: white; }
            """
        )

    def choose_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите Excel-файл", "", "Excel (*.xlsx *.xls)")
        if path:
            self.excel_path_edit.setText(path)
            self._reset_state_for_new_file()

    def _reset_state_for_new_file(self) -> None:
        self.current_data = None
        self.equipment_catalogs_cleared = False
        self.baseline = None
        self.results_by_algorithm.clear()
        self.run_data_by_algorithm.clear()
        self.last_algorithm = None
        self.graphs_tab.clear_figures()
        self.tables_tab.clear_tables()
        self.economics_tab.clear_results()
        self.load_output.clear()
        for edit in self.algorithm_outputs.values():
            edit.clear()

    def _excel_path(self) -> str:
        path = self.excel_path_edit.text().strip()
        if not path:
            raise ValueError("Не выбран Excel-файл с исходными данными.")
        return path

    def check_loading(self) -> None:
        try:
            excel_path = Path(self._excel_path())
            if self._uses_custom_equipment():
                data = self._load_data_with_current_equipment(excel_path)
            else:
                data = load_source_data(excel_path)
            self.current_data = data
            self._show_data_status("Исходный файл исправен, правильно загружен и готов к работе")
        except Exception as exc:
            self.load_output.setPlainText("Исходный файл загружен не верно\n\n" + str(exc))
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))

    def run_optimization(self) -> None:
        algorithm = self.algorithm_combo.currentText()
        self.algorithm_outputs[algorithm].setPlainText(
            f"Запущена оптимизация по алгоритму {algorithm}. Подождите завершения расчета..."
        )

        if self._uses_custom_equipment():
            try:
                data, results_dir = self._prepare_custom_data_for_calculation()
            except Exception as exc:
                QMessageBox.warning(self, "Недостаточно данных", str(exc))
                return
            self._set_busy(True)
            self.worker = CustomEquipmentOptimizationWorker(data, results_dir, algorithm, self.fuel_price_spin.value())
        else:
            try:
                excel_path = self._excel_path()
            except Exception as exc:
                QMessageBox.warning(self, "Не выбран файл", str(exc))
                return
            self._set_busy(True)
            self.worker = OptimizationWorker(excel_path, algorithm, self.fuel_price_spin.value())

        self.worker.finished_ok.connect(self.on_optimization_ok)
        self.worker.failed.connect(self.on_optimization_failed)
        self.worker.start()

    def compare_algorithms(self) -> None:
        for algorithm in ALGORITHMS:
            self.algorithm_outputs[algorithm].setPlainText(
                f"Запущено сравнение. Алгоритм {algorithm} будет рассчитан по очереди..."
            )

        if self._uses_custom_equipment():
            try:
                data, results_dir = self._prepare_custom_data_for_calculation()
            except Exception as exc:
                QMessageBox.warning(self, "Недостаточно данных", str(exc))
                return
            self._set_busy(True)
            self.worker = CustomEquipmentCompareWorker(data, results_dir, self.fuel_price_spin.value())
        else:
            try:
                excel_path = self._excel_path()
            except Exception as exc:
                QMessageBox.warning(self, "Не выбран файл", str(exc))
                return
            self._set_busy(True)
            self.worker = CompareWorker(excel_path, self.fuel_price_spin.value())

        self.worker.finished_ok.connect(self.on_compare_ok)
        self.worker.failed.connect(self.on_optimization_failed)
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.load_btn.setEnabled(not busy)
        self.compare_btn.setEnabled(not busy)
        self.manual_equipment_btn.setEnabled(not busy)
        self.clear_equipment_btn.setEnabled(not busy)
        self.open_graph_btn.setEnabled(not busy)
        self.open_all_graphs_btn.setEnabled(not busy)
        self.open_table_btn.setEnabled(not busy)
        self.open_all_tables_btn.setEnabled(not busy)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

    # -------------------------- графики и таблицы --------------------------

    def open_selected_graph(self) -> None:
        key = self.graph_combo.currentData()
        errors = []
        for algorithm in self._target_algorithms():
            try:
                self._open_graph_for_algorithm(algorithm, key)
            except Exception as exc:
                errors.append(f"{algorithm} / {self.graph_combo.currentText()}: {exc}")
        if errors:
            QMessageBox.information(self, "Часть графиков недоступна", "\n".join(errors))
        self.main_tabs.setCurrentWidget(self.graphs_tab)

    def open_all_graphs(self) -> None:
        errors = []
        for algorithm in self._target_algorithms():
            for key, title in GRAPH_CHOICES:
                try:
                    self._open_graph_for_algorithm(algorithm, key)
                except Exception as exc:
                    errors.append(f"{algorithm} / {title}: {exc}")
        if errors:
            QMessageBox.information(self, "Часть графиков недоступна", "\n".join(errors))
        self.main_tabs.setCurrentWidget(self.graphs_tab)

    def _open_graph_for_algorithm(self, algorithm: str, key: str) -> None:
        if self.current_data is None:
            raise ValueError("Сначала выполните проверку загрузки исходного файла или оптимизацию.")
        title = dict(GRAPH_CHOICES)[key]
        if key == "existing_year":
            canvas = build_existing_year_figure(self.current_data)
        else:
            result = self.results_by_algorithm.get(algorithm)
            if result is None:
                raise ValueError("нет результата оптимизации")
            if key == "final_year":
                canvas = build_final_year_figure(result.best, algorithm=algorithm)
            elif key == "jan15":
                canvas = build_final_day_figure(result.best, month=1, day=15, algorithm=algorithm)
            elif key == "apr15":
                canvas = build_final_day_figure(result.best, month=4, day=15, algorithm=algorithm)
            elif key == "jul15":
                canvas = build_final_day_figure(result.best, month=7, day=15, algorithm=algorithm)
            elif key == "oct15":
                canvas = build_final_day_figure(result.best, month=10, day=15, algorithm=algorithm)
            else:
                raise ValueError("неизвестный тип графика")
        self.graphs_tab.open_canvas(algorithm, title, canvas)

    def open_selected_table(self) -> None:
        key = self.table_combo.currentData()
        errors = []
        for algorithm in self._target_algorithms():
            try:
                self._open_table_for_algorithm(algorithm, key)
            except Exception as exc:
                errors.append(f"{algorithm} / {self.table_combo.currentText()}: {exc}")
        if errors:
            QMessageBox.information(self, "Часть таблиц недоступна", "\n".join(errors))
        self.main_tabs.setCurrentWidget(self.tables_tab)

    def open_all_tables(self) -> None:
        errors = []
        for algorithm in self._target_algorithms():
            for key, title in TABLE_CHOICES:
                try:
                    self._open_table_for_algorithm(algorithm, key)
                except Exception as exc:
                    errors.append(f"{algorithm} / {title}: {exc}")
        if errors:
            QMessageBox.information(self, "Часть таблиц недоступна", "\n".join(errors))
        self.main_tabs.setCurrentWidget(self.tables_tab)

    def _open_table_for_algorithm(self, algorithm: str, key: str) -> None:
        if self.current_data is None:
            raise ValueError("Сначала выполните проверку загрузки исходного файла или оптимизацию.")
        title = dict(TABLE_CHOICES)[key]
        if key == "existing_year":
            df = build_existing_year_table(self.current_data)
        else:
            result = self.results_by_algorithm.get(algorithm)
            if result is None:
                raise ValueError("нет результата оптимизации")
            if key == "final_year":
                df = build_final_year_table(result.best)
            elif key == "jan15":
                df = build_final_day_table(result.best, month=1, day=15)
            elif key == "apr15":
                df = build_final_day_table(result.best, month=4, day=15)
            elif key == "jul15":
                df = build_final_day_table(result.best, month=7, day=15)
            elif key == "oct15":
                df = build_final_day_table(result.best, month=10, day=15)
            else:
                raise ValueError("неизвестный тип таблицы")
        self.tables_tab.open_table(algorithm, title, df)

    def _target_algorithms(self) -> list[str]:
        target = self.target_algorithm_combo.currentText()
        if target == "Все алгоритмы":
            return ALGORITHMS.copy()
        if target == "Последний рассчитанный":
            if self.last_algorithm is not None:
                return [self.last_algorithm]
            return [self.algorithm_combo.currentText()]
        return [target]

    def _open_default_outputs_for_algorithm(self, algorithm: str) -> None:
        for key, _ in GRAPH_CHOICES:
            try:
                self._open_graph_for_algorithm(algorithm, key)
            except Exception:
                pass
        for key, _ in TABLE_CHOICES:
            try:
                self._open_table_for_algorithm(algorithm, key)
            except Exception:
                pass

    # -------------------------- результаты --------------------------

    def on_optimization_ok(self, algorithm: str, text: str, data: object, result: object, baseline: object) -> None:
        self._set_busy(False)
        if isinstance(data, SourceData):
            self.current_data = data
            self.load_output.setPlainText(
                "Исходный файл исправен, правильно загружен и готов к работе\n\n"
                + equipment_catalog_summary(data)
            )
        if isinstance(result, OptimizationResult) and isinstance(baseline, SimulationResult):
            self.results_by_algorithm[algorithm] = result
            self.baseline = baseline
            self.run_data_by_algorithm[algorithm] = RunData(result=result, baseline=baseline)
            self.last_algorithm = algorithm
            self.algorithm_outputs[algorithm].setPlainText(text)
            self.graphs_tab.clear_figures(algorithm)
            self.tables_tab.clear_tables(algorithm)
            self._open_default_outputs_for_algorithm(algorithm)
            self.economics_tab.set_results(self.run_data_by_algorithm, self.fuel_price_spin.value())

    def on_compare_ok(self, texts: object, data: object, results: object, baseline: object) -> None:
        self._set_busy(False)
        if isinstance(data, SourceData):
            self.current_data = data
            self.load_output.setPlainText(
                "Исходный файл исправен, правильно загружен и готов к работе\n\n"
                + equipment_catalog_summary(data)
            )
        if isinstance(results, dict) and isinstance(baseline, SimulationResult):
            self.baseline = baseline
            for algorithm, result in results.items():
                if isinstance(result, OptimizationResult):
                    self.results_by_algorithm[algorithm] = result
                    self.run_data_by_algorithm[algorithm] = RunData(result=result, baseline=baseline)
                    self.last_algorithm = algorithm
                    self.graphs_tab.clear_figures(algorithm)
                    self.tables_tab.clear_tables(algorithm)
                    self._open_default_outputs_for_algorithm(algorithm)
            if isinstance(texts, dict):
                for algorithm, text in texts.items():
                    if algorithm in self.algorithm_outputs:
                        self.algorithm_outputs[algorithm].setPlainText(str(text))
            self.economics_tab.set_results(self.run_data_by_algorithm, self.fuel_price_spin.value())

    def on_optimization_failed(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Ошибка оптимизации", message)


def _run_single_calculation(excel_path: str, algorithm: str) -> tuple[SourceData, SimulationResult, OptimizationResult]:
    data = load_source_data(excel_path)
    config = OptimizationConfig()
    baseline = _simulate_baseline(data, config)
    result = run_optimization(data, algorithm=algorithm, config=config)
    return data, baseline, result


def _run_custom_calculation(data: SourceData, algorithm: str) -> tuple[SourceData, SimulationResult, OptimizationResult]:
    config = OptimizationConfig()
    baseline = _simulate_baseline(data, config)
    result = run_optimization(data, algorithm=algorithm, config=config)
    return data, baseline, result


def _simulate_baseline(data: SourceData, config: OptimizationConfig) -> SimulationResult:
    baseline_choice = EquipmentChoice(
        wind_index=0,
        wind_count=0,
        battery_index=0,
        battery_target_kwh=0.0,
        inverter_index=0,
        inverter_count=0,
    )
    return simulate_system(data, baseline_choice, config.system, make_hourly=False)


def _save_result_files(excel_path: str, algorithm: str, result: OptimizationResult) -> None:
    try:
        base_dir = Path(excel_path).parent
        hourly_path = base_dir / f"results_hourly_{algorithm}.csv"
        monthly_path = base_dir / f"results_monthly_{algorithm}.csv"
        result.best.hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
        from energy_models import monthly_summary

        monthly_summary(result.best).to_csv(monthly_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _save_result_files_to_dir(results_dir: str | Path, algorithm: str, result: OptimizationResult) -> None:
    try:
        base_dir = Path(results_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        hourly_path = base_dir / f"results_hourly_{algorithm}.csv"
        monthly_path = base_dir / f"results_monthly_{algorithm}.csv"
        result.best.hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
        from energy_models import monthly_summary

        monthly_summary(result.best).to_csv(monthly_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _append_equipment_rows(base: pd.DataFrame, rows: pd.DataFrame, empty_factory) -> pd.DataFrame:
    if base is None or base.empty:
        result = empty_factory()
    else:
        result = base.copy()

    if rows is None or rows.empty:
        return result.reset_index(drop=True)

    add = rows.copy()
    for column in result.columns:
        if column not in add.columns:
            add[column] = 0.0 if column != "model" else ""
    add = add[result.columns]

    if result.empty:
        next_id = 1
    else:
        next_id = int(pd.to_numeric(result["id"], errors="coerce").fillna(0).max()) + 1
    add["id"] = range(next_id, next_id + len(add))
    return pd.concat([result, add], ignore_index=True).reset_index(drop=True)


def _ensure_equipment_ready(data: SourceData) -> None:
    missing = []
    if data.wind_turbines.empty:
        missing.append("ВЭУ")
    if data.batteries.empty:
        missing.append("АКБ")
    if data.inverters.empty:
        missing.append("инверторы")
    if missing:
        raise ValueError(
            "Для оптимизации не заполнены каталоги оборудования: "
            + ", ".join(missing)
            + ". Загрузите Excel с каталогами или добавьте оборудование вручную."
        )

    checks = [
        (data.wind_turbines, "ВЭУ", ["rated_power_kw", "hub_height_m", "price_rub"]),
        (data.batteries, "АКБ", ["voltage_v", "capacity_ah", "energy_kwh", "price_rub"]),
        (data.inverters, "Инверторы", ["rated_power_kw", "max_power_kw", "peak_power_kw", "price_rub"]),
    ]
    errors = []
    for df, name, columns in checks:
        for column in columns:
            if column not in df.columns:
                errors.append(f"{name}: отсутствует столбец {column}")
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            if values.isna().any():
                errors.append(f"{name}: есть пустые или некорректные значения в {column}")
            if (values < 0).any():
                errors.append(f"{name}: есть отрицательные значения в {column}")
    if errors:
        raise ValueError("Некорректные параметры оборудования:\n" + "\n".join(errors))


def manual_equipment_summary(wind: pd.DataFrame, batteries: pd.DataFrame, inverters: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"Пользовательские ВЭУ: {len(wind)} строк",
            f"Пользовательские АКБ: {len(batteries)} строк",
            f"Пользовательские инверторы: {len(inverters)} строк",
        ]
    )


def equipment_catalog_summary(data: SourceData) -> str:
    return "\n".join(
        [
            f"Каталог ВЭУ: {len(data.wind_turbines)} строк",
            f"Каталог АКБ: {len(data.batteries)} строк",
            f"Каталог инверторов: {len(data.inverters)} строк",
        ]
    )


def format_optimization_report(result: OptimizationResult, baseline: SimulationResult, fuel_price_rub_l: float) -> str:
    best = result.best
    e = best.equipment
    m = best.metrics
    base_m = baseline.metrics

    before_fuel = float(base_m["diesel_fuel_l"])
    after_fuel = float(m["diesel_fuel_l"])
    fuel_saving = before_fuel - after_fuel
    fuel_saving_percent = fuel_saving / before_fuel * 100.0 if before_fuel > 1e-9 else 0.0

    before_cost = before_fuel * fuel_price_rub_l
    after_cost = after_fuel * fuel_price_rub_l
    cost_saving = before_cost - after_cost

    lines = [
        f"Алгоритм: {result.algorithm}",
        f"Количество расчетов режима: {_fmt_number(result.evaluations, 0)}",
        f"Время расчета: {_fmt_number(result.elapsed_seconds, 2)} с",
        "",
        "Выбранное оборудование:",
        f"  ВЭУ: {e.wind.get('model', '-')} x {e.wind.get('count', 0)} шт., Pуст = {_fmt_number(e.wind.get('installed_power_kw', 0), 1)} кВт",
        f"  АКБ: {e.battery.get('model', '-')} x {_fmt_number(e.battery.get('total_cells', 0), 0)} элементов; схема {e.battery.get('series_count', 0)}s x {e.battery.get('parallel_count', 0)}p; E = {_fmt_number(e.battery.get('energy_total_kwh', 0), 1)} кВт·ч",
        f"  Инвертор: {e.inverter.get('model', '-')} x {e.inverter.get('count', 0)} шт., Pном = {_fmt_number(e.inverter.get('rated_power_total_kw', 0), 1)} кВт",
        f"  Стоимость оборудования: {_fmt_number(e.total_price_rub, 0)} руб",
        "",
        "Результаты режима:",
        f"  Расход топлива ДЭС после оптимизации: {_fmt_number(after_fuel, 1)} л",
        f"  Выработка ДЭС после оптимизации: {_fmt_number(m['diesel_energy_kwh'], 1)} кВт·ч",
        f"  Использовано ВЭУ: {_fmt_number(m['wind_used_kwh'], 1)} кВт·ч из {_fmt_number(m['wind_available_kwh'], 1)} кВт·ч (Kиспл = {_fmt_number(m['wind_utilization_fraction'] * 100, 2)} %)",
        f"  Непокрытая нагрузка: {_fmt_number(m['unserved_kwh'], 1)} кВт·ч",
        f"  Балластная нагрузка после оптимизации: {_fmt_number(m.get('ballast_kwh', 0.0), 1)} кВт·ч",
        "",
        "Результаты оптимизации:",
        f"  Расход топлива ДЭС до оптимизации: {_fmt_number(before_fuel, 1)} л",
        f"  Расход топлива ДЭС после оптимизации: {_fmt_number(after_fuel, 1)} л",
        f"  Снижение расхода топлива: {_fmt_number(fuel_saving, 1)} л ({_fmt_number(fuel_saving_percent, 2)} %)",
        f"  Цена дизельного топлива: {_fmt_number(fuel_price_rub_l, 2)} руб/л",
        f"  Стоимость топлива ДЭС до оптимизации: {_fmt_number(before_cost, 1)} руб",
        f"  Стоимость топлива ДЭС после оптимизации: {_fmt_number(after_cost, 1)} руб",
        f"  Экономия на топливе: {_fmt_number(cost_saving, 1)} руб",
    ]

    if e.violations:
        lines.append("")
        lines.append("Ограничения/предупреждения:")
        lines.extend(f"  - {v}" for v in e.violations)

    return "\n".join(lines)


def _fmt_number(value: object, decimals: int = 1) -> str:
    try:
        return f"{float(value):,.{decimals}f}".replace(",", " ").replace(".", ",")
    except Exception:
        return str(value)


def _group_with_text(title: str, widget: QPlainTextEdit) -> QGroupBox:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.addWidget(widget)
    return group


def main() -> None:
    apply_windows_taskbar_icon()

    app = QApplication(sys.argv)
    app.setWindowIcon(get_app_icon())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
