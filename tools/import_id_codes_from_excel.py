import argparse
import re
from pathlib import Path

import pandas as pd
import pymysql


def _safe_table_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("table name must be string")
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise ValueError("table name contains invalid characters")
    return name


def _find_id_column(columns) -> str:
    # Priority candidates (case-insensitive)
    candidates = [
        "id_code", "idcode", "id code", "code", "id",
        "二维码", "二维码号", "二维码编号", "笼号", "笼位", "编号",
    ]
    cols = [str(c).strip() for c in columns]
    lower_map = {c.lower(): c for c in cols}
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    # Fallback: first column
    return cols[0] if cols else ""


def _normalize_code(val) -> str:
    if val is None:
        return ""
    try:
        if isinstance(val, float):
            if val.is_integer():
                return str(int(val))
        if isinstance(val, int):
            return str(val)
    except Exception:
        pass
    s = str(val).strip()
    if s.endswith(".0"):
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return s


def _ensure_table(cursor, table: str) -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        id_code VARCHAR(64) NOT NULL,
        img_path VARCHAR(512),
        scan_time DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cursor.execute(sql)
    try:
        cursor.execute(f"CREATE INDEX idx_{table}_id_code ON {table}(id_code)")
    except Exception:
        pass


def _load_codes(excel_path: Path, sheet_prefix: str, sheet_count: int) -> list[str]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    xls = pd.ExcelFile(excel_path)
    codes = []
    for i in range(1, sheet_count + 1):
        sheet_name = f"{sheet_prefix}{i:02d}"
        if sheet_name not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty:
            continue
        id_col = _find_id_column(df.columns)
        if not id_col or id_col not in df.columns:
            continue
        for v in df[id_col].tolist():
            code = _normalize_code(v)
            if code:
                codes.append(code)
    # Dedupe while preserving order
    seen = set()
    uniq = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def _fetch_existing_ids(cursor, table: str) -> set:
    existing = set()
    cursor.execute(f"SELECT id_code FROM {table}")
    for row in cursor.fetchall():
        if row and row[0]:
            existing.add(str(row[0]).strip())
    return existing


def main():
    parser = argparse.ArgumentParser(description="Import id_code from Excel tables into MySQL.")
    parser.add_argument("--excel", default=r"C:\Users\Administrator\Desktop\江州打印终表.xlsx", help="Excel file path")
    parser.add_argument("--sheet-prefix", default="Table_", help="Sheet name prefix")
    parser.add_argument("--sheet-count", type=int, default=15, help="How many sheets to scan")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--db", default="wenshi_eggs_record")
    parser.add_argument("--table", default="qr_code_images")
    parser.add_argument("--ensure-table", action="store_true", default=True)
    parser.add_argument("--dedupe-db", action="store_true", default=True)
    args = parser.parse_args()

    excel_path = Path(args.excel)
    table = _safe_table_name(args.table)

    codes = _load_codes(excel_path, args.sheet_prefix, args.sheet_count)
    if not codes:
        print("No id_code found.")
        return

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        db=args.db,
        charset="utf8mb4",
    )
    cursor = conn.cursor()

    if args.ensure_table:
        _ensure_table(cursor, table)
        conn.commit()

    if args.dedupe_db:
        existing = _fetch_existing_ids(cursor, table)
        codes = [c for c in codes if c not in existing]

    if not codes:
        print("No new id_code to insert.")
        return

    sql = f"INSERT INTO {table} (id_code) VALUES (%s)"
    cursor.executemany(sql, [(c,) for c in codes])
    conn.commit()
    print(f"Inserted {len(codes)} id_code into {table}.")


if __name__ == "__main__":
    main()
