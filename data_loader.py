from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


class DataLoadError(Exception):
    pass


@dataclass
class SourceData:
    load_year: pd.DataFrame
    load_hour: pd.DataFrame
    mges_year: pd.DataFrame
    mges_hour: pd.DataFrame
    des_year: pd.DataFrame
    wind_hour: pd.DataFrame
    wind_turbines: pd.DataFrame
    batteries: pd.DataFrame
    inverters: pd.DataFrame

    def summary(self) -> str:
        rows = [
            f"Годовая нагрузка: {len(self.load_year)} строк",
            f"Почасовая нагрузка: {len(self.load_hour)} строк",
            f"Годовая выработка МГЭС: {len(self.mges_year)} строк",
            f"Почасовая выработка МГЭС: {len(self.mges_hour)} строк",
            f"Годовая выработка ДЭС: {len(self.des_year)} строк",
            f"Почасовой ветер: {len(self.wind_hour)} строк",
            f"Каталог ВЭУ: {len(self.wind_turbines)} строк",
            f"Каталог АКБ: {len(self.batteries)} строк",
            f"Каталог инверторов: {len(self.inverters)} строк",
        ]
        return "\n".join(rows)


SHEETS = {
    "load": "Нагрузка",
    "mges": "МГЭС",
    "des": "ДЭС",
    "wind": "Ветер",
    "wind_turbines": "ВЭУ",
    "batteries": "АКБ",
    "inverters": "Инверторы",
}

MONTHS_RU = {
    "Январь": 1,
    "Февраль": 2,
    "Март": 3,
    "Апрель": 4,
    "Май": 5,
    "Июнь": 6,
    "Июль": 7,
    "Август": 8,
    "Сентябрь": 9,
    "Октябрь": 10,
    "Ноябрь": 11,
    "Декабрь": 12,
}


def load_source_data(excel_path: str | Path) -> SourceData:
    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise DataLoadError(f"Файл не найден: {excel_path}")

    _check_required_sheets(excel_path)

    load_year = _load_load_year(excel_path)
    load_hour = _load_load_hour(excel_path)

    mges_year = _load_mges_year(excel_path)
    mges_hour = _load_mges_hour(excel_path)

    des_year = _load_des_year(excel_path)
    wind_hour = _load_wind_hour(excel_path)

    wind_turbines = _load_wind_turbines(excel_path)
    batteries = _load_batteries(excel_path)
    inverters = _load_inverters(excel_path)

    data = SourceData(
        load_year=load_year,
        load_hour=load_hour,
        mges_year=mges_year,
        mges_hour=mges_hour,
        des_year=des_year,
        wind_hour=wind_hour,
        wind_turbines=wind_turbines,
        batteries=batteries,
        inverters=inverters,
    )

    validate_source_data(data)
    return data


def _check_required_sheets(excel_path: Path) -> None:
    actual_sheets = set(pd.ExcelFile(excel_path).sheet_names)
    missing = [name for name in SHEETS.values() if name not in actual_sheets]

    if missing:
        raise DataLoadError(
            "В Excel отсутствуют обязательные листы: " + ", ".join(missing)
        )


def _load_load_year(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["load"],
        usecols="A:C",
        skiprows=1,
        nrows=12,
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Номер": "month_number",
            "Месяц": "month_name",
            "Wпотр.мес, кВт*ч": "load_kwh",
        }
    )

    df["month_number"] = _to_int(df["month_number"])
    df["month_name"] = df["month_name"].astype(str).str.strip()
    df["load_kwh"] = _to_float(df["load_kwh"])
    return df[["month_number", "month_name", "load_kwh"]]


def _load_load_hour(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["load"],
        usecols="E:I",
        skiprows=1,
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Год": "year",
            "Месяц.1": "month",
            "Месяц": "month",
            "День": "day",
            "Час": "hour",
            "Wпотр.час, кВт*ч": "load_kwh",
        }
    )

    df = _clean_hourly_datetime_frame(df)
    df["load_kwh"] = _to_float(df["load_kwh"])
    return df[["datetime", "year", "month", "day", "hour", "load_kwh"]]


def _load_mges_year(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["mges"],
        usecols="A:D",
        skiprows=1,
        nrows=12,
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Номер": "month_number",
            "Месяц": "month_name",
            "Wмгэс1, кВт*ч": "mges1_kwh",
            "Wмгэс2, кВт*ч": "mges2_kwh",
        }
    )

    df["month_number"] = _to_int(df["month_number"])
    df["month_name"] = df["month_name"].astype(str).str.strip()
    df["mges1_kwh"] = _to_float(df["mges1_kwh"])
    df["mges2_kwh"] = _to_float(df["mges2_kwh"])
    df["mges_total_kwh"] = df["mges1_kwh"] + df["mges2_kwh"]
    return df[["month_number", "month_name", "mges1_kwh", "mges2_kwh", "mges_total_kwh"]]


def _load_mges_hour(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["mges"],
        usecols="F:K",
        skiprows=1,
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Год": "year",
            "Месяц.1": "month",
            "Месяц": "month",
            "День": "day",
            "Час": "hour",
            "Wмгэс1, кВт*ч.1": "mges1_kwh",
            "Wмгэс2, кВт*ч.1": "mges2_kwh",
            "Wмгэс1, кВт*ч": "mges1_kwh",
            "Wмгэс2, кВт*ч": "mges2_kwh",
        }
    )

    df = _clean_hourly_datetime_frame(df)
    df["mges1_kwh"] = _to_float(df["mges1_kwh"])
    df["mges2_kwh"] = _to_float(df["mges2_kwh"])
    df["mges_total_kwh"] = df["mges1_kwh"] + df["mges2_kwh"]
    return df[["datetime", "year", "month", "day", "hour", "mges1_kwh", "mges2_kwh", "mges_total_kwh"]]


def _load_des_year(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["des"],
        usecols="A:C",
        skiprows=1,
        nrows=12,
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Номер": "month_number",
            "Месяц": "month_name",
            "Wдэс, кВт*ч": "des_kwh",
        }
    )

    df["month_number"] = _to_int(df["month_number"])
    df["month_name"] = df["month_name"].astype(str).str.strip()
    df["des_kwh"] = _to_float(df["des_kwh"])
    return df[["month_number", "month_name", "des_kwh"]]


def _load_wind_hour(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["wind"],
        usecols="A:E",
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Год": "year",
            "Месяц": "month",
            "День": "day",
            "Час": "hour",
            "Скорость ветра на высоте 10м, м/с": "wind_speed_10m_mps",
        }
    )

    df = _clean_hourly_datetime_frame(df)
    df["wind_speed_10m_mps"] = _to_float(df["wind_speed_10m_mps"])
    return df[["datetime", "year", "month", "day", "hour", "wind_speed_10m_mps"]]


def _load_wind_turbines(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["wind_turbines"],
        usecols="A:E",
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Номер": "id",
            "Марка": "model",
            "Pном, кВт": "rated_power_kw",
            "Высота оси (h), м": "hub_height_m",
            "Цена, руб": "price_rub",
        }
    )

    df["id"] = _to_int(df["id"])
    df["model"] = df["model"].astype(str).str.strip()
    df["rated_power_kw"] = _to_float(df["rated_power_kw"])
    df["hub_height_m"] = _to_float(df["hub_height_m"])
    df["price_rub"] = _to_float(df["price_rub"])

    curves = _load_wind_power_curves(excel_path)
    df["power_curve_speeds_mps"] = df["id"].map(lambda i: curves.get(int(i), {}).get("speeds", tuple()))
    df["power_curve_powers_kw"] = df["id"].map(lambda i: curves.get(int(i), {}).get("powers", tuple()))
    df["has_power_curve"] = df["power_curve_speeds_mps"].map(lambda values: len(values) > 0)

    return df[
        [
            "id",
            "model",
            "rated_power_kw",
            "hub_height_m",
            "price_rub",
            "power_curve_speeds_mps",
            "power_curve_powers_kw",
            "has_power_curve",
        ]
    ]


def _load_wind_power_curves(excel_path: Path) -> dict[int, dict[str, tuple[float, ...]]]:
    try:
        raw = pd.read_excel(
            excel_path,
            sheet_name=SHEETS["wind_turbines"],
            header=None,
        )
    except Exception:
        return {}

    if raw.empty or raw.shape[1] <= 5:
        return {}

    curve_columns: list[tuple[int, int]] = []
    for col in range(5, raw.shape[1]):
        turbine_id = _extract_int_from_cell(raw.iat[0, col])
        if turbine_id is not None and 1 <= turbine_id <= 500:
            curve_columns.append((turbine_id, col))

    if not curve_columns:
        return {}

    first_curve_col = min(col for _, col in curve_columns)
    speed_col = _find_wind_curve_speed_column(raw, first_curve_col)
    if speed_col is None:
        return {}

    speed_rows: list[tuple[int, float]] = []
    for row in range(1, raw.shape[0]):
        speed = _extract_float_from_cell(raw.iat[row, speed_col])
        if speed is not None and 0.0 <= speed <= 80.0:
            speed_rows.append((row, speed))

    if len(speed_rows) < 2:
        return {}

    curves: dict[int, dict[str, tuple[float, ...]]] = {}
    for turbine_id, col in curve_columns:
        speeds: list[float] = []
        powers: list[float] = []
        for row, speed in speed_rows:
            power = _extract_float_from_cell(raw.iat[row, col])
            if power is None:
                continue
            speeds.append(float(speed))
            powers.append(max(0.0, float(power)))

        if len(speeds) >= 2:
            pairs = sorted(zip(speeds, powers), key=lambda item: item[0])
            curves[int(turbine_id)] = {
                "speeds": tuple(float(s) for s, _ in pairs),
                "powers": tuple(float(p) for _, p in pairs),
            }

    return curves


def _find_wind_curve_speed_column(raw: pd.DataFrame, first_curve_col: int) -> int | None:
    best_col: int | None = None
    best_count = 0
    for col in range(max(0, first_curve_col - 3), first_curve_col):
        count = 0
        for row in range(1, raw.shape[0]):
            value = raw.iat[row, col]
            speed = _extract_float_from_cell(value)
            if speed is not None and 0.0 <= speed <= 80.0:
                # Предпочитаем именно ячейки со скоростями, где часто есть текст м/с.
                text = str(value).lower()
                if "м/с" in text or "m/s" in text or speed <= 30.0:
                    count += 1
        if count > best_count:
            best_count = count
            best_col = col
    return best_col if best_count >= 2 else None


def _extract_float_from_cell(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_int_from_cell(value: object) -> int | None:
    number = _extract_float_from_cell(value)
    if number is None:
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9:
        return None
    return rounded


def _load_batteries(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["batteries"],
        usecols="A:H",
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Номер": "id",
            "Марка": "model",
            "Uном, В": "voltage_v",
            "Cном, Ач": "capacity_ah",
            "Уд. Энергия, Вт*ч/кг": "specific_energy_wh_kg",
            "Уд. Энергия, ВТ*ч/л": "specific_energy_wh_l",
            "Масса, кг": "mass_kg",
            "Цена, руб": "price_rub",
        }
    )

    df["id"] = _to_int(df["id"])
    df["model"] = df["model"].astype(str).str.strip()
    df["voltage_v"] = _to_float(df["voltage_v"])
    df["capacity_ah"] = _to_float(df["capacity_ah"])
    df["specific_energy_wh_kg"] = _to_float(df["specific_energy_wh_kg"])
    df["specific_energy_wh_l"] = _to_float(df["specific_energy_wh_l"])
    df["mass_kg"] = _to_float(df["mass_kg"])
    df["price_rub"] = _to_float(df["price_rub"])
    df["energy_kwh"] = df["voltage_v"] * df["capacity_ah"] / 1000.0

    return df[
        [
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
    ]


def _load_inverters(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name=SHEETS["inverters"],
        usecols="A:I",
    )

    df = _drop_empty_rows(df)
    df = df.rename(
        columns={
            "Номер": "id",
            "Марка": "model",
            "Pном, кВт": "rated_power_kw",
            "Pmax, кВт": "max_power_kw",
            "Pпик, кВт": "peak_power_kw",
            "Uвх, В": "input_voltage_v",
            "Uвых, В": "output_voltage_v",
            "Cmax, Ач": "max_capacity_ah",
            "Цена, руб": "price_rub",
        }
    )

    df["id"] = _to_int(df["id"])
    df["model"] = df["model"].astype(str).str.strip()
    df["rated_power_kw"] = _to_float(df["rated_power_kw"])
    df["max_power_kw"] = _to_float(df["max_power_kw"])
    df["peak_power_kw"] = _to_float(df["peak_power_kw"])
    df["input_voltage_v"] = _to_float(df["input_voltage_v"])
    df["output_voltage_v"] = _to_float(df["output_voltage_v"])
    df["max_capacity_ah"] = _to_float(df["max_capacity_ah"])
    df["price_rub"] = _to_float(df["price_rub"])

    return df[
        [
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
    ]


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)
    return df.reset_index(drop=True)


def _clean_hourly_datetime_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["year", "month", "day", "hour"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise DataLoadError(
            "В почасовой таблице отсутствуют колонки: " + ", ".join(missing)
        )

    df = df.copy()
    df["year"] = _to_int(df["year"])
    df["month"] = _to_int(df["month"])
    df["day"] = _to_int(df["day"])
    df["hour"] = _to_int(df["hour"])

    df["datetime"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
        },
        errors="raise",
    )

    return df.sort_values("datetime").reset_index(drop=True)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="raise",
    ).astype(float)


def _to_int(series: pd.Series) -> pd.Series:
    return _to_float(series).round().astype(int)


def validate_source_data(data: SourceData) -> None:
    _validate_monthly_table(data.load_year, "Годовая нагрузка", ["load_kwh"])
    _validate_monthly_table(data.mges_year, "Годовая выработка МГЭС", ["mges1_kwh", "mges2_kwh", "mges_total_kwh"])
    _validate_monthly_table(data.des_year, "Годовая выработка ДЭС", ["des_kwh"])

    _validate_hourly_table(data.load_hour, "Почасовая нагрузка", ["load_kwh"])
    _validate_hourly_table(data.mges_hour, "Почасовая выработка МГЭС", ["mges1_kwh", "mges2_kwh", "mges_total_kwh"])
    _validate_hourly_table(data.wind_hour, "Почасовая скорость ветра", ["wind_speed_10m_mps"])

    _validate_same_timeline(data.load_hour, data.mges_hour, "нагрузка", "МГЭС")
    _validate_same_timeline(data.load_hour, data.wind_hour, "нагрузка", "ветер")

    _validate_equipment(data.wind_turbines, "ВЭУ", ["rated_power_kw", "hub_height_m", "price_rub"])
    _validate_equipment(data.batteries, "АКБ", ["voltage_v", "capacity_ah", "energy_kwh", "price_rub"])
    _validate_equipment(data.inverters, "Инверторы", ["rated_power_kw", "max_power_kw", "peak_power_kw", "price_rub"])


def _validate_monthly_table(df: pd.DataFrame, table_name: str, numeric_columns: list[str]) -> None:
    if len(df) != 12:
        raise DataLoadError(f"{table_name}: должно быть 12 строк по месяцам, получено {len(df)}")

    expected = list(range(1, 13))
    actual = df["month_number"].tolist()

    if actual != expected:
        raise DataLoadError(f"{table_name}: номера месяцев должны быть от 1 до 12")

    _check_non_negative(df, table_name, numeric_columns)


def _validate_hourly_table(df: pd.DataFrame, table_name: str, numeric_columns: list[str]) -> None:
    rows_count = len(df)

    if rows_count not in (8760, 8784):
        raise DataLoadError(
            f"{table_name}: ожидается 8760 строк для обычного года или 8784 для високосного, получено {rows_count}"
        )

    if df["datetime"].duplicated().any():
        raise DataLoadError(f"{table_name}: найдены повторяющиеся дата и час")

    if not df["datetime"].is_monotonic_increasing:
        raise DataLoadError(f"{table_name}: дата и час должны идти по порядку")

    _check_non_negative(df, table_name, numeric_columns)


def _validate_same_timeline(base: pd.DataFrame, other: pd.DataFrame, base_name: str, other_name: str) -> None:
    if len(base) != len(other):
        raise DataLoadError(
            f"Почасовые ряды не совпадают по длине: {base_name}={len(base)}, {other_name}={len(other)}"
        )

    if not base["datetime"].equals(other["datetime"]):
        raise DataLoadError(f"Почасовые ряды {base_name} и {other_name} не совпадают по датам и часам")


def _validate_equipment(df: pd.DataFrame, table_name: str, numeric_columns: list[str]) -> None:
    if df.empty:
        raise DataLoadError(f"Каталог оборудования {table_name} пустой")

    if df["id"].duplicated().any():
        raise DataLoadError(f"Каталог оборудования {table_name}: есть повторяющиеся номера")

    if df["model"].astype(str).str.strip().eq("").any():
        raise DataLoadError(f"Каталог оборудования {table_name}: есть пустые марки оборудования")

    _check_non_negative(df, table_name, numeric_columns)


def _check_non_negative(df: pd.DataFrame, table_name: str, numeric_columns: list[str]) -> None:
    for column in numeric_columns:
        if column not in df.columns:
            raise DataLoadError(f"{table_name}: отсутствует колонка {column}")

        if df[column].isna().any():
            raise DataLoadError(f"{table_name}: колонка {column} содержит пустые значения")

        if (df[column] < 0).any():
            raise DataLoadError(f"{table_name}: колонка {column} содержит отрицательные значения")


if __name__ == "__main__":
    data = load_source_data("Исходные данные.xlsx")
    print(data.summary())
    print("\nПервые строки почасовой нагрузки:")
    print(data.load_hour.head())
