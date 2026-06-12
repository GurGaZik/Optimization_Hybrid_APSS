from data_loader import load_source_data
from optimizer import run_optimization

EXCEL_PATH = r"Исходные данные.xlsx"
ALGORITHM = "Greedy"

data = load_source_data(EXCEL_PATH)
print(data.summary())
print()

result = run_optimization(data, algorithm=ALGORITHM)
print(result.report())
