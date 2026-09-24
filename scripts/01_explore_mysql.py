# -*- coding: utf-8 -*-
"""
01 数据探查：从 MySQL 读回 9 张表，打印概况
目的：治理之前，先看清每张表的规模、列类型、空值、数值分布。

用法：
    $env:MYSQL_PASSWORD='你的密码'
    python scripts/01_explore_mysql.py
"""
import os
import pandas as pd
import pymysql

TABLES = [
    "customers", "orders", "order_items", "payments", "products",
    "reviews", "sellers", "geolocation", "product_category_name_translation",
]


def main():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database="olist",
        charset="utf8mb4",
    )

    for t in TABLES:
        df = pd.read_sql(f"SELECT * FROM `{t}`", conn)
        print("=" * 78)
        print(f"表: {t}    行数: {len(df):,}    列: {df.shape[1]}")
        print("-" * 78)

        # 每列：类型 + 空值数
        for col in df.columns:
            nulls = int(df[col].isna().sum())
            print(f"  {col:<32} {str(df[col].dtype):<12} 空值: {nulls:,}")

        # 数值列：最小值/中位数/最大值（一眼看出是否合理）
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            print("  --- 数值列分布(min / 中位数 / max) ---")
            for c in num_cols:
                s = df[c]
                print(f"  {c:<32} {s.min():,.2f} / {s.median():,.2f} / {s.max():,.2f}")

        # 日期列：范围
        date_cols = [c for c in df.columns if "date" in c.lower() or "timestamp" in c.lower()]
        for c in date_cols:
            s = pd.to_datetime(df[c], errors="coerce").dropna()
            if len(s) > 0:
                print(f"  {c:<32} {s.min()} ~ {s.max()}")

    conn.close()
    print("\n探查完成。")


if __name__ == "__main__":
    main()
