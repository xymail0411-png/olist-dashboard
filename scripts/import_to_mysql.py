# -*- coding: utf-8 -*-
"""
CSV -> MySQL 导入脚本（Olist 9 张表）

用法：
    $env:MYSQL_PASSWORD='你的密码'
    python scripts/import_to_mysql.py

导入顺序按外键依赖排列：先导无依赖的表，再导被引用的表。
"""
import os
import sys

import pandas as pd
import pymysql

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "olist")

DB_CONFIG = dict(
    host="localhost",
    user="root",
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database="olist",
    charset="utf8mb4",
)

# (MySQL 表名, CSV 文件名) —— 顺序即导入顺序，不能乱
TABLES = [
    ("customers", "olist_customers_dataset.csv"),
    ("products", "olist_products_dataset.csv"),
    ("sellers", "olist_sellers_dataset.csv"),
    ("geolocation", "olist_geolocation_dataset.csv"),
    ("product_category_name_translation", "product_category_name_translation.csv"),
    ("orders", "olist_orders_dataset.csv"),
    ("order_items", "olist_order_items_dataset.csv"),
    ("payments", "olist_order_payments_dataset.csv"),
    ("reviews", "olist_order_reviews_dataset.csv"),
]

BATCH_SIZE = 20000  # 每批插入行数（geolocation 有 100 万行，分批防内存/超时）


def to_sql_value(v):
    """pandas 的空值 -> SQL NULL，其余原样（保持字符串，MySQL 自动转换类型）"""
    if pd.isna(v):
        return None
    return v


def main():
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    for table, csv_file in TABLES:
        path = os.path.join(DATA_DIR, csv_file)
        df = pd.read_csv(path)

        cols = df.columns.tolist()
        col_str = ",".join(f"`{c}`" for c in cols)
        placeholders = ",".join(["%s"] * len(cols))
        sql = f"INSERT INTO `{table}` ({col_str}) VALUES ({placeholders})"

        rows = [tuple(to_sql_value(v) for v in r) for r in df.itertuples(index=False)]

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            cur.executemany(sql, batch)
            conn.commit()

        print(f"[OK] {table}: 插入 {len(rows):,} 行 <- {csv_file}")

    cur.close()
    conn.close()
    print("\n全部导入完成。下一步：SELECT COUNT(*) 逐表核对行数。")


if __name__ == "__main__":
    main()
