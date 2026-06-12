from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QTabBar, QTabWidget, QVBoxLayout, QWidget

from data_loader import SourceData
from energy_models import SimulationResult


MONTH_LABELS = {
    1: "Янв",
    2: "Фев",
    3: "Мар",
    4: "Апр",
    5: "Май",
    6: "Июн",
    7: "Июл",
    8: "Авг",
    9: "Сен",
    10: "Окт",
    11: "Ноя",
    12: "Дек",
}

MONTH_GENITIVE = {
    1: "января",
    4: "апреля",
    7: "июля",
    10: "октября",
}

PLOT_COLORS = {
    "mges1": "#5dade2",       # голубой
    "mges2": "#1f4e9a",       # синий
    "wind": "#2ca02c",        # зеленый
    "diesel1": "#d62728",     # красный
    "diesel2": "#800020",     # бордовый
    "charge": "#8b5a2b",      # коричневый
    "discharge": "#9467bd",   # фиолетовый
    "ballast": "#7f7f7f",     # серый
    "load": "#111111",
}

ALGORITHMS = ["Greedy", "GWO", "DE"]


class GraphsPanel(QWidget):

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        self.open_figures: dict[str, dict[str, FigureCanvas]] = {
            algorithm: {} for algorithm in ALGORITHMS
        }

        export_row = QHBoxLayout()
        self.export_current_btn = QPushButton("Скачать текущий график")
        self.export_current_btn.clicked.connect(self.export_current_figure)
        self.export_all_btn = QPushButton("Скачать все графики")
        self.export_all_btn.clicked.connect(self.export_all_figures)
        export_row.addWidget(self.export_current_btn)
        export_row.addWidget(self.export_all_btn)
        export_row.addStretch(1)
        outer.addLayout(export_row)

        self.algorithm_tabs = QTabWidget()
        self.inner_tabs: dict[str, QTabWidget] = {}
        for algorithm in ALGORITHMS:
            inner = QTabWidget()
            inner.setDocumentMode(False)
            inner.setMovable(True)
            inner.setTabsClosable(False)
            inner.setStyleSheet(_tab_close_style())
            self.inner_tabs[algorithm] = inner
            self.algorithm_tabs.addTab(inner, algorithm)
        outer.addWidget(self.algorithm_tabs, 1)

    def clear_figures(self, algorithm: str | None = None) -> None:
        targets = [algorithm] if algorithm else ALGORITHMS
        for alg in targets:
            self.open_figures.setdefault(alg, {}).clear()
            inner = self.inner_tabs.get(alg)
            if inner is None:
                continue
            while inner.count():
                widget = inner.widget(0)
                inner.removeTab(0)
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

    def open_canvas(self, algorithm: str, title: str, canvas: FigureCanvas) -> None:
        inner = self.inner_tabs.get(algorithm)
        if inner is None:
            raise ValueError(f"Неизвестный алгоритм для вкладки графиков: {algorithm}")

        existing_index = _find_tab(inner, title)
        if existing_index >= 0:
            _close_tab(inner, existing_index)

        self.open_figures.setdefault(algorithm, {})[title] = canvas

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(canvas, 1)

        index = inner.addTab(page, title)
        _install_close_button(inner, index)
        inner.setCurrentIndex(index)
        self.algorithm_tabs.setCurrentWidget(inner)

    def export_current_figure(self) -> None:
        algorithm = self.algorithm_tabs.tabText(self.algorithm_tabs.currentIndex())
        inner = self.inner_tabs.get(algorithm)

        if inner is None or inner.currentIndex() < 0:
            QMessageBox.information(self, "Экспорт графика", "Нет открытого графика для экспорта.")
            return

        title = inner.tabText(inner.currentIndex())
        canvas = self.open_figures.get(algorithm, {}).get(title)

        if canvas is None:
            QMessageBox.information(self, "Экспорт графика", "Данные текущего графика не найдены.")
            return

        default_name = _safe_filename(f"{algorithm}_{title}.png")
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить график",
            default_name,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;JPEG (*.jpg *.jpeg)",
        )

        if not path:
            return

        _save_figure(canvas, path, selected_filter)
        QMessageBox.information(self, "Экспорт графика", "График сохранен.")

    def export_all_figures(self) -> None:
        figures: list[tuple[str, str, FigureCanvas]] = []
        for algorithm, inner in self.inner_tabs.items():
            for idx in range(inner.count()):
                title = inner.tabText(idx)
                canvas = self.open_figures.get(algorithm, {}).get(title)
                if canvas is not None:
                    figures.append((algorithm, title, canvas))

        if not figures:
            QMessageBox.information(self, "Экспорт графиков", "Нет открытых графиков для экспорта.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения графиков")
        if not folder:
            return

        folder_path = Path(folder)
        for algorithm, title, canvas in figures:
            filename = _safe_filename(f"{algorithm}_{title}.png")
            _save_figure(canvas, str(folder_path / filename), "PNG (*.png)")

        QMessageBox.information(self, "Экспорт графиков", "Все открытые графики сохранены в PNG.")


def build_existing_year_figure(data: SourceData) -> FigureCanvas:
    df = _prepare_existing_year_data(data)
    figure = _make_figure(figsize=(11.3, 6.2))
    ax = figure.add_subplot(111)

    x = np.arange(len(df))
    labels = df["label"].tolist()

    bottom = np.zeros(len(df))
    bottom = _stack_bar(ax, x, df["mges1_kwh"].to_numpy(), bottom, "МГЭС1", PLOT_COLORS["mges1"])
    bottom = _stack_bar(ax, x, df["mges2_kwh"].to_numpy(), bottom, "МГЭС2", PLOT_COLORS["mges2"])
    bottom = _stack_bar(ax, x, df["diesel1_kwh"].to_numpy(), bottom, "ДЭУ1", PLOT_COLORS["diesel1"])
    bottom = _stack_bar(ax, x, df["diesel2_kwh"].to_numpy(), bottom, "ДЭУ2", PLOT_COLORS["diesel2"])

    _plot_load_line(ax, x, df["load_kwh"].to_numpy())
    _format_axes(
        ax,
        labels,
        "Годовой график распределения нагрузки (существующая схема)",
        "Энергия (кВт·ч)",
    )
    figure.tight_layout(pad=2.0)
    return FigureCanvas(figure)


def build_final_year_figure(result: SimulationResult, algorithm: str | None = None) -> FigureCanvas:
    df = _prepare_final_year_data(result)
    figure = _make_figure(figsize=(11.3, 6.4))
    ax = figure.add_subplot(111)

    x = np.arange(len(df))
    labels = df["label"].tolist()

    bottom = np.zeros(len(df))
    bottom = _stack_bar(ax, x, df["mges1_kwh"].to_numpy(), bottom, "МГЭС1", PLOT_COLORS["mges1"])
    bottom = _stack_bar(ax, x, df["mges2_kwh"].to_numpy(), bottom, "МГЭС2", PLOT_COLORS["mges2"])
    bottom = _stack_bar(ax, x, df["wind_used_kwh"].to_numpy(), bottom, "ВЭУ", PLOT_COLORS["wind"])
    bottom = _stack_bar(ax, x, df["diesel1_kwh"].to_numpy(), bottom, "ДЭУ1", PLOT_COLORS["diesel1"])
    bottom = _stack_bar(ax, x, df["diesel2_kwh"].to_numpy(), bottom, "ДЭУ2", PLOT_COLORS["diesel2"])
    bottom = _stack_bar(ax, x, df["battery_discharge_kwh"].to_numpy(), bottom, "Разряд АКБ", PLOT_COLORS["discharge"])
    bottom = _stack_bar(ax, x, df["battery_charge_kwh"].to_numpy(), bottom, "Заряд АКБ", PLOT_COLORS["charge"])
    _negative_bar(ax, x, df["ballast_kwh"].to_numpy(), "Балластная нагрузка", PLOT_COLORS["ballast"])

    _plot_load_line(ax, x, df["load_kwh"].to_numpy())
    alg = algorithm or "выбранному алгоритму"
    _format_axes(
        ax,
        labels,
        f"Годовой график распределения нагрузки (оптимизированная схема по {alg})",
        "Энергия (кВт·ч)",
        allow_negative=True,
    )
    figure.tight_layout(pad=2.0)
    return FigureCanvas(figure)


def build_final_day_figure(result: SimulationResult, month: int, day: int = 15, algorithm: str | None = None) -> FigureCanvas:
    df = _prepare_final_day_data(result, month=month, day=day)
    figure = _make_figure(figsize=(11.3, 6.4))
    ax = figure.add_subplot(111)

    x = np.arange(len(df))
    labels = [str(int(h)) for h in df["hour"].tolist()]

    bottom = np.zeros(len(df))
    bottom = _stack_bar(ax, x, df["mges1_kwh"].to_numpy(), bottom, "МГЭС1", PLOT_COLORS["mges1"])
    bottom = _stack_bar(ax, x, df["mges2_kwh"].to_numpy(), bottom, "МГЭС2", PLOT_COLORS["mges2"])
    bottom = _stack_bar(ax, x, df["wind_used_kwh"].to_numpy(), bottom, "ВЭУ", PLOT_COLORS["wind"])
    bottom = _stack_bar(ax, x, df["diesel1_kwh"].to_numpy(), bottom, "ДЭУ1", PLOT_COLORS["diesel1"])
    bottom = _stack_bar(ax, x, df["diesel2_kwh"].to_numpy(), bottom, "ДЭУ2", PLOT_COLORS["diesel2"])
    bottom = _stack_bar(ax, x, df["battery_discharge_kwh"].to_numpy(), bottom, "Разряд АКБ", PLOT_COLORS["discharge"])
    bottom = _stack_bar(ax, x, df["battery_charge_kwh"].to_numpy(), bottom, "Заряд АКБ", PLOT_COLORS["charge"])
    _negative_bar(ax, x, df["ballast_kwh"].to_numpy(), "Балластная нагрузка", PLOT_COLORS["ballast"])

    _plot_load_line(ax, x, df["load_kwh"].to_numpy())
    alg = algorithm or "выбранному алгоритму"
    month_name = MONTH_GENITIVE.get(month, f"{month} месяца")
    _format_axes(
        ax,
        labels,
        f"Суточный график распределения нагрузки для {day} {month_name} (оптимизированная схема по {alg})",
        "Энергия (кВт·ч)",
        x_label="Время (часы)",
        allow_negative=True,
    )
    figure.tight_layout(pad=2.0)
    return FigureCanvas(figure)


def _prepare_existing_year_data(data: SourceData) -> pd.DataFrame:
    load_df = data.load_year.copy()
    mges_df = data.mges_year.copy()
    des_df = data.des_year.copy()

    df = load_df.merge(mges_df[["month_number", "mges1_kwh", "mges2_kwh"]], on="month_number", how="left")
    df = df.merge(des_df[["month_number", "des_kwh"]], on="month_number", how="left")
    df["des_kwh"] = df["des_kwh"].fillna(0.0)
    df["mges1_kwh"] = df["mges1_kwh"].fillna(0.0)
    df["mges2_kwh"] = df["mges2_kwh"].fillna(0.0)
    df["diesel1_kwh"] = df["des_kwh"] / 2.0
    df["diesel2_kwh"] = df["des_kwh"] / 2.0
    df["label"] = df["month_number"].map(MONTH_LABELS)
    return df[["month_number", "label", "load_kwh", "mges1_kwh", "mges2_kwh", "diesel1_kwh", "diesel2_kwh"]]

def _prepare_final_year_data(result: SimulationResult) -> pd.DataFrame:
    df = _with_diesel_split_columns(result.hourly)
    dt = pd.to_datetime(df["datetime"])
    df = df.assign(month=dt.dt.month)
    cols = [
        "load_kwh",
        "mges1_kwh",
        "mges2_kwh",
        "wind_used_kwh",
        "diesel1_kwh",
        "diesel2_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "ballast_kwh",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    grouped = df.groupby("month", as_index=False)[cols].sum()
    grouped["label"] = grouped["month"].map(MONTH_LABELS)
    return grouped

def _prepare_final_day_data(result: SimulationResult, month: int, day: int = 15) -> pd.DataFrame:
    df = _with_diesel_split_columns(result.hourly)
    dt = pd.to_datetime(df["datetime"])
    mask = (dt.dt.month == int(month)) & (dt.dt.day == int(day))
    day_df = df.loc[mask].copy()
    if day_df.empty:
        raise ValueError(f"В результатах нет данных за {day:02d}.{month:02d}.")
    if "ballast_kwh" not in day_df.columns:
        day_df["ballast_kwh"] = 0.0
    day_df["hour"] = pd.to_datetime(day_df["datetime"]).dt.hour
    day_df = day_df.sort_values("hour")
    return day_df[
        [
            "hour",
            "load_kwh",
            "mges1_kwh",
            "mges2_kwh",
            "wind_used_kwh",
            "diesel1_kwh",
            "diesel2_kwh",
            "battery_charge_kwh",
            "battery_discharge_kwh",
            "ballast_kwh",
        ]
    ]


def _with_diesel_split_columns(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    diesel_units = df["diesel_units"].fillna(0).astype(int)
    per_unit = df["diesel_per_unit_kw"].fillna(0.0).astype(float)
    df["diesel1_kwh"] = np.where(diesel_units >= 1, per_unit, 0.0)
    df["diesel2_kwh"] = np.where(diesel_units >= 2, per_unit, 0.0)
    return df

def _make_figure(figsize: tuple[float, float]) -> Figure:
    return Figure(figsize=figsize, dpi=110)


def _stack_bar(ax, x: np.ndarray, values: np.ndarray, bottom: np.ndarray, label: str, color: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ax.bar(x, values, bottom=bottom, width=0.82, label=label, color=color, edgecolor="black", linewidth=0.35)
    return bottom + values


def _negative_bar(ax, x: np.ndarray, values: np.ndarray, label: str, color: str) -> None:
    values = np.asarray(values, dtype=float)
    if np.nanmax(np.abs(values)) > 1e-9:
        ax.bar(x, -values, bottom=0.0, width=0.82, label=label, color=color, edgecolor="black", linewidth=0.35)


def _plot_load_line(ax, x: np.ndarray, load: np.ndarray) -> None:
    ax.plot(x, load, color=PLOT_COLORS["load"], marker="o", markersize=3.2, linewidth=2.0, label="Нагрузка", zorder=5)


def _format_axes(ax, x_labels: list[str], title: str, y_label: str, x_label: str = "Месяц", allow_negative: bool = False) -> None:
    ax.set_title(title, fontsize=10.5, pad=16)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)

    ymax = 0.0
    ymin = 0.0
    for patch in ax.patches:
        y0 = patch.get_y()
        y1 = patch.get_y() + patch.get_height()
        ymax = max(ymax, y0, y1)
        ymin = min(ymin, y0, y1)
    for line in ax.lines:
        if len(line.get_ydata()):
            ymax = max(ymax, float(np.nanmax(line.get_ydata())))
    if allow_negative and ymin < 0:
        ax.set_ylim(ymin * 1.20, ymax * 1.12 if ymax > 0 else 1.0)
        ax.axhline(0, color="#333333", linewidth=0.8)
    else:
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8.5,
    )


def _save_figure(canvas: FigureCanvas, path: str, selected_filter: str = "") -> None:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if not suffix:
        if "PDF" in selected_filter:
            path_obj = path_obj.with_suffix(".pdf")
        elif "SVG" in selected_filter:
            path_obj = path_obj.with_suffix(".svg")
        elif "JPEG" in selected_filter:
            path_obj = path_obj.with_suffix(".jpg")
        else:
            path_obj = path_obj.with_suffix(".png")

    canvas.figure.savefig(path_obj, dpi=220, bbox_inches="tight")


def _safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] or "graph.png"


def _find_tab(tab_widget: QTabWidget, title: str) -> int:
    for i in range(tab_widget.count()):
        if tab_widget.tabText(i) == title:
            return i
    return -1


def _close_tab(tab_widget: QTabWidget, index: int) -> None:
    widget = tab_widget.widget(index)
    tab_widget.removeTab(index)
    if widget is not None:
        widget.setParent(None)
        widget.deleteLater()


def _install_close_button(tab_widget: QTabWidget, index: int) -> None:
    button = QPushButton("×")
    button.setFixedSize(24, 24)
    button.setToolTip("Закрыть вкладку")
    button.setStyleSheet(
        "QPushButton { color: black; font-size: 18px; font-weight: bold; border: none; background: transparent; }"
        "QPushButton:hover { background: #dddddd; border-radius: 11px; }"
    )
    button.clicked.connect(lambda _checked=False, tw=tab_widget, page=tab_widget.widget(index): _close_tab_by_widget(tw, page))
    position = QTabBar.ButtonPosition.RightSide
    tab_widget.tabBar().setTabButton(index, position, button)


def _close_tab_by_widget(tab_widget: QTabWidget, page: QWidget) -> None:
    index = tab_widget.indexOf(page)
    if index >= 0:
        _close_tab(tab_widget, index)


def _tab_close_style() -> str:
    return """
    QTabBar::tab { padding: 7px 34px 7px 10px; }
    """
