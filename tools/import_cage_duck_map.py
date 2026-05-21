import os
import sys
import pandas as pd
import pymysql


EXCEL_PATH = r"C:\Users\12179\Desktop\江州打印终表.xlsx"
DB_CFG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "db": "wenshi_eggs_record",
    "charset": "utf8mb4",
}
TABLE_NAME = "cage_duck_map"


def parse_id_code(id_code: str):
    if not isinstance(id_code, str):
        return None, None
    s = id_code.strip()
    if not s:
        return None, None
    if "/" not in s:
        return None, None
    cage_str, cx_str = s.split("/", 1)
    try:
        cage = int(cage_str)
    except Exception:
        cage = None
    try:
        cx_wb = int(cx_str)
    except Exception:
        cx_wb = None
    return cage, cx_wb


def build_records():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel不存在: {EXCEL_PATH}")

    xls = pd.ExcelFile(EXCEL_PATH)
    records = []
    order_index = 1
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        if df.empty:
            continue
        # 第一行是 A8_value，跳过
        values = df.iloc[1:, 0].tolist()
        row_no = 1
        for val in values:
            if pd.isna(val):
                row_no += 1
                continue
            id_code = str(val).strip()
            if not id_code:
                row_no += 1
                continue
            cage, cx_wb = parse_id_code(id_code)
            records.append({
                "id_code": id_code,
                "cx_wb": cx_wb,
                "cage": cage,
                "sheet_name": sheet_name,
                "row_no": row_no,
                "order_index": order_index,
            })
            order_index += 1
            row_no += 1
    return records


def main():
    records = build_records()
    if not records:
        print("未解析到任何有效的id_code。")
        return

    conn = pymysql.connect(**DB_CFG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {TABLE_NAME}")
            sql = (
                f"INSERT INTO {TABLE_NAME} "
                "(id_code, cx_wb, cage, sheet_name, row_no, order_index) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            )
            values = [
                (r["id_code"], r["cx_wb"], r["cage"], r["sheet_name"], r["row_no"], r["order_index"])
                for r in records
            ]
            cursor.executemany(sql, values)
        conn.commit()
    finally:
        conn.close()

    print(f"导入完成：{len(records)} 条")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"导入失败: {e}")
        sys.exit(1)
