# -*- coding: utf-8 -*-
"""生成图3：复购客 vs 单次客 画像差异（平均品类数 / Top1品类占比 / 评分均值）
用新数据（2017-01 起）重算，替换 docs/figures/profile_single_vs_repeat.png
数据源：orders_master / customer_rfm(stage) / order_items / products_clean
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "olist"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
C1, C2 = "#C9D0D9", "#52647D"   # 单次(浅) / 复购(深)


def main():
    master = pd.read_csv(PROCESSED / "orders_master.csv")
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    prods = pd.read_csv(PROCESSED / "products_clean.csv")[["product_id", "product_category_name_english"]]

    ci = items.merge(prods, on="product_id", how="left").merge(
        master[["order_id", "customer_unique_id"]], on="order_id", how="left")
    ci["cat"] = ci["product_category_name_english"].fillna("unknown")

    # 每客户品类数
    ncat = ci.groupby("customer_unique_id")["cat"].nunique().rename("平均品类数")
    # 每客户 Top1 品类占比（出现最多的品类 / 总品类次数）
    cnt = ci.groupby(["customer_unique_id", "cat"]).size().rename("n").reset_index()
    per = cnt.groupby("customer_unique_id")["n"].agg(lambda s: s.max() / s.sum()).rename("Top1品类占比")
    # 每客户评分均值
    score = master[master["review_score"] > 0].groupby("customer_unique_id")["review_score"].mean().rename("评分均值")

    cust = rfm[["customer_unique_id", "stage"]].merge(ncat, on="customer_unique_id", how="left") \
        .merge(per, on="customer_unique_id", how="left").merge(score, on="customer_unique_id", how="left")

    single = cust[cust["stage"] == "单次购买"]
    repeat = cust[cust["stage"] == "复购客户"]
    print(f"单次 {len(single):,} ｜ 复购 {len(repeat):,}")
    metrics = [("平均品类数", "平均品类数"), ("Top1品类占比", "Top1品类占比"), ("评分均值", "评分均值")]
    vals = {m: (single[c].mean(), repeat[c].mean()) for m, c in metrics}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    labels = ["单次客", "复购客"]
    for ax, (title, col) in zip(axes, metrics):
        s, r = vals[col]
        bars = ax.bar(labels, [s, r], color=[C1, C2], width=0.5)
        ax.bar_label(bars, fmt="%.2f" if col != "评分均值" else "%.2f", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#E2E2E2", linewidth=.5)
        ax.set_axisbelow(True)
        ax.margins(y=.25)
    fig.suptitle("复购客 vs 单次客：品类多样性与体验差异", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "profile_single_vs_repeat.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已保存: docs/figures/profile_single_vs_repeat.png")


if __name__ == "__main__":
    main()
