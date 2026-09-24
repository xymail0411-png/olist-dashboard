# -*- coding: utf-8 -*-
"""
09 增长分解 + cohort 留存：平台靠拉新还是留旧（结构说的直接量化）
数据源：
  - data/processed/orders_master.csv   订单主宽表（order_purchase_timestamp / total_value / customer_unique_id）
产出：
  - data/processed/growth_decomposition.csv  月度 GMV 构成（新客首单 vs 老客复购）
  - data/processed/cohort_retention.csv       cohort 留存率矩阵
  - docs/figures/growth_gmv_split.png         月度 GMV 构成堆叠图
  - docs/figures/growth_cohort.png            cohort 留存曲线/热力图

设计要点（重要）：
  1) 新客首单 = 客户第 1 笔订单（order_rank==1）；老客复购 = 第 2+ 笔（order_rank>=2）
  2) cohort = 客户首购月份；留存率 = 某 cohort 在后续各月的活跃客户数 / 首月客户数
  3) 回答主线：GMV 靠新客还是老客？留存率低 => 一次性买家主导（结构说）
用法：
    C:\\...\\python.exe scripts/09_growth_decomposition.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

NEW, OLD = "#52647D", "#C9D0D9"    # 新客首单(藏青) / 老客复购(浅)


def main():
    master = pd.read_csv(PROCESSED / "orders_master.csv",
                         parse_dates=["order_purchase_timestamp"]).sort_values("order_purchase_timestamp")

    master["ym"] = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
    # 每客户第几单（已按时间排序）
    master["order_rank"] = master.groupby("customer_unique_id").cumcount() + 1
    # 客户首购月份 = cohort
    master["cohort_ym"] = master.groupby("customer_unique_id")["ym"].transform("min")
    master["单类型"] = np.where(master["order_rank"] == 1, "新客首单", "老客复购")

    # ============ 1. 月度 GMV 构成分解 ============
    gmv = master.groupby(["ym", "单类型"])["total_value"].sum().unstack().fillna(0)
    # 月度新客数 / 复购单数
    new_cust_month = master[master["单类型"] == "新客首单"].groupby("ym")["customer_unique_id"].nunique()
    old_orders_month = master[master["单类型"] == "老客复购"].groupby("ym").size()

    gmv = gmv.join(new_cust_month.rename("新客数"), how="left").join(old_orders_month.rename("复购单数"), how="left")
    gmv["新客首单占比%"] = (gmv["新客首单"] / (gmv["新客首单"] + gmv["老客复购"]) * 100).round(1)
    gmv = gmv.reset_index()
    gmv.to_csv(PROCESSED / "growth_decomposition.csv", index=False, encoding="utf-8-sig")

    total_new = gmv["新客首单"].sum()
    total_old = gmv["老客复购"].sum()
    print("=" * 60)
    print("月度 GMV 构成（新客首单 vs 老客复购），累计：")
    print(f"  新客首单 GMV {total_new:,.0f}（{total_new/(total_new+total_old)*100:.1f}%）｜ 老客复购 GMV {total_old:,.0f}（{total_old/(total_new+total_old)*100:.1f}%）")
    print(f"  两年累计新客数 {new_cust_month.sum():,.0f}，复购订单数 {old_orders_month.sum():,.0f}")

    # ============ 2. cohort 留存率 ============
    active = master.groupby(["cohort_ym", "ym"])["customer_unique_id"].nunique().unstack()
    # 严格口径：cohort 规模 = 该群组「首月」（对角线）客户数；留存率 = 某月活跃数 / 首月客户数
    cohort_size = pd.Series({c: (active.loc[c, c] if c in active.columns else np.nan) for c in active.index})
    retention = active.div(cohort_size, axis=0).round(3)
    retention.to_csv(PROCESSED / "cohort_retention.csv", encoding="utf-8-sig")

    # 计算各 cohort 第 1/3/6/12 个月留存
    print("\ncohort 留存率抽样（首月=100%）：")
    for c in retention.index[:6]:
        row = retention.loc[c]
        later = row[row.index > c]
        if len(later):
            m3 = later.iloc[min(2, len(later)-1)] if len(later) >= 3 else later.iloc[-1]
            print(f"  cohort {c}: 首月客户 {int(cohort_size[c]):,}，第3个月留存 {later.iloc[min(2,len(later)-1)]*100:.1f}%")

    print("\n已保存: data/processed/growth_decomposition.csv / cohort_retention.csv")

    # ============ 图1 月度 GMV 构成堆叠图 ============
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(gmv))
    b1 = ax.bar(x, gmv["新客首单"], color=NEW, label="新客首单 GMV", width=.7)
    b2 = ax.bar(x, gmv["老客复购"], bottom=gmv["新客首单"], color=OLD, label="老客复购 GMV", width=.7)
    total_m = (gmv["新客首单"] + gmv["老客复购"]) / 1e6
    ax.bar_label(b2, labels=[f"{t:.2f}" for t in total_m], fontsize=6.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(gmv["ym"], rotation=45, fontsize=8)
    ax.set_ylabel("GMV (R$)")
    ax.set_title("月度 GMV 构成：新客首单 vs 老客复购（柱顶=总 GMV，单位百万 R$）")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    ax.margins(y=.12)
    fig.tight_layout()
    fig.savefig(FIG / "growth_gmv_split.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============ 图2 cohort 留存热力图 ============
    # 只取前 12 个 cohort、前 12 个月
    r = retention.iloc[:12, :12] * 100
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(r.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(8, r.values.max()))
    ax.set_xticks(range(r.shape[1]))
    ax.set_xticklabels(r.columns, rotation=45, fontsize=7)
    ax.set_yticks(range(r.shape[0]))
    ax.set_yticklabels(r.index, fontsize=8)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 4 else "#252525")
    ax.set_xlabel("订单月份")
    ax.set_ylabel("首购月份（cohort）")
    ax.set_title("Cohort 留存率热力图（%）——对角线为首购月=100%")
    fig.tight_layout()
    fig.savefig(FIG / "growth_cohort.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已保存: docs/figures/growth_gmv_split.png / growth_cohort.png")


if __name__ == "__main__":
    main()
