import os
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

_SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def is_spreadsheet_file(file_name: str) -> bool:
    return os.path.splitext(file_name.lower())[1] in _SPREADSHEET_EXTENSIONS


async def load_spreadsheet_tables(
    file_path: str,
    file_name: str,
) -> Tuple[List[Tuple[str, pd.DataFrame]], Optional[str]]:
    ext = os.path.splitext(file_name.lower())[1]
    if ext not in _SPREADSHEET_EXTENSIONS:
        return [], f"{file_name} is not spreadsheet format"
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
            return [("sheet_0_data", df)], None
        xls = pd.ExcelFile(file_path)
        tables: List[Tuple[str, pd.DataFrame]] = []
        for i, sheet in enumerate(xls.sheet_names):
            safe = "".join(ch if ch.isalnum() else "_" for ch in sheet.lower()).strip("_")
            safe = safe or f"sheet_{i}"
            df = pd.read_excel(file_path, sheet_name=sheet)
            tables.append((f"sheet_{i}_{safe}", df))
        return tables, None
    except Exception as e:
        return [], str(e)


def build_schema_description(tables: List[Tuple[str, pd.DataFrame]]) -> str:
    lines: List[str] = []
    for table_name, df in tables:
        cols = ", ".join(f"{c}:{str(t).upper()}" for c, t in zip(df.columns, df.dtypes))
        lines.append(f"Table: {table_name} ({cols})")
    return "\n".join(lines)


def execute_sql_on_sheets(
    tables: List[Tuple[str, pd.DataFrame]],
    sql: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    con = duckdb.connect(":memory:")
    try:
        for table_name, df in tables:
            con.register(table_name, df)
        result = con.execute(sql)
        columns = [d[0] for d in result.description]
        rows = [dict(zip(columns, r)) for r in result.fetchall()]
        return rows, None
    except Exception as e:
        return [], str(e)
    finally:
        con.close()

