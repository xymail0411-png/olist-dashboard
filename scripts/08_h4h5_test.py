# -*- coding: utf-8 -*-
"""
08 H4/H5 履约归因：超时为什么发生、哪部分可改
数据源：
  - data/processed/orders_master.csv         订单主宽表（is_late 超时 / estimated_days 预计 / delivery_days 实际 / customer_state）
  - data/olist/olist_order_items_dataset.csv 订单明细（order_id -> seller_id / product_id）
  - data/olist/olist_sellers_dataset.csv     卖家（seller_id -> seller_state）
  - data/processed/products_clean.csv        商品类目（product_id -> product_category_name_english）
产出：
  - data/processed/h4h5_results.csv          各维度检验结果汇总
  - docs/figures/h4_cross_state.png          同州 vs 跨州 超时率
  - docs/figures/h5_state.png                客户州 超时率 top
  - docs/figures/h5_category.png             品类 超时率 top
  - docs/figures/h5_seller.png               卖家 承诺过紧 vs 物流延误

H4 表述：超时与跨州距离相关，订单集中于东南部。
H5 表述：超时集中在特定州、品类、卖家，可区分"承诺过紧"与"物流延误"。

设计要点（重要）：
  1) 订单级超时（is_late）由 orders_master 提供；一单可能多商品/多卖家，
     取每单第一个商品（order_item_id=1）的卖家州与品类作为该单代表。
  2) H4：客户州 vs 卖家州 是否跨州 -> 超时率（若跨州显著更高 => 结构性距离因素）
  3) H5：
       州/品类/卖家维度超时率拆解，找超时集中点；
       承诺过紧 vs 物流延误：卖家级对比 estimated_days(预计) vs delivery_days(实际)，
       实际不慢却超时多 = 承诺过紧(可改)；实际确实慢 = 物流延误(难改)。
用法：
    C:\\...\\python.exe scripts/08_h4h5_test.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "olist"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2 = "#C9D0D9", "#52647D"
BAD, OK = "#B84C3A", "#52647D"


def eff_label_v(v):
    return "大" if v > .3 else "中" if v > .1 else "小"


def cramers_v(chi2, n, r, k):
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def main():
    master = pd.read_csv(PROCESSED / "orders_master.csv",
                         parse_dates=["order_purchase_timestamp"]).sort_values("order_purchase_timestamp")
    master["is_late"] = master["is_late"].astype(float)   # bool 列 merge 后易变 object，先转 float
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
    prods = pd.read_csv(PROCESSED / "products_clean.csv")[["product_id", "product_category_name_english"]]

    # 订单主卖家/主品类（每单第一个商品）
    first_item = (items.sort_values("order_item_id")
                  .groupby("order_id").first().reset_index())
    first_item = first_item.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    first_item = first_item.merge(prods, on="product_id", how="left")
    m = master.merge(first_item[["order_id", "seller_state", "product_category_name_english"]],
                     on="order_id", how="left")

    rows = []

    # ============ H4 跨州 vs 同州 ============
    valid = m[m["seller_state"].notna()].copy()
    valid["跨州"] = valid["customer_state"] != valid["seller_state"]
    cross = valid[valid["跨州"]]
    same = valid[~valid["跨州"]]
    cr = cross["is_late"].mean()
    sr = same["is_late"].mean()
    ct = pd.crosstab(valid["跨州"], valid["is_late"])
    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
    print(f"[H4 跨州] 同州超时率 {sr*100:.2f}% (n={len(same):,}) vs 跨州 {cr*100:.2f}% (n={len(cross):,}) | V={v:.3f}")
    rows.append({"假设": "H4", "维度": "同州 vs 跨州", "同州超时率%": round(sr*100, 2), "跨州超时率%": round(cr*100, 2),
                 "跨州订单占比%": round(len(cross)/len(valid)*100, 1), "卡方p": p, "效应量V": v,
                 "结论": "跨州显著拉高超时率" if v > .1 and cr > sr else "跨州对超时影响弱"})

    # ============ H5 州 / 品类 / 卖家维度超时率 ============
    # 客户州
    st = m.groupby("customer_state")["is_late"].agg(["mean", "count"]).reset_index()
    st = st[st["count"] >= 200].sort_values("mean", ascending=False)
    st["超时率%"] = (st["mean"]*100).round(2)
    print("\n[H5 客户州] 超时率最高前5州：")
    print(st.head(5)[["customer_state", "超时率%", "count"]].to_string(index=False))
    rows.append({"假设": "H5", "维度": "客户州top", "明细": st.head(5).to_dict("records")})

    # 品类
    cat = m[m["product_category_name_english"].notna()].groupby("product_category_name_english")["is_late"].agg(["mean", "count"]).reset_index()
    cat = cat[cat["count"] >= 300].sort_values("mean", ascending=False)
    cat["超时率%"] = (cat["mean"]*100).round(2)
    print("\n[H5 品类] 超时率最高前5品类：")
    print(cat.head(5)[["product_category_name_english", "超时率%", "count"]].to_string(index=False))

    # 卖家：超时率 + 平均预计/实际天数（承诺过紧 vs 物流延误）
    # 卖家超时率用明细粒度（一单多卖家各算，is_late 为订单级重复）
    si = items.merge(master[["order_id", "is_late", "delivery_days", "estimated_days"]], on="order_id", how="left")
    sell = si.groupby("seller_id")["is_late"].agg(["mean", "count"]).reset_index()
    sell = sell[sell["count"] >= 30].copy()          # 只看有足够单量的卖家
    sell["超时率%"] = (sell["mean"]*100).round(2)
    # 卖家平均预计/实际天数
    sdur = si.groupby("seller_id")[["estimated_days", "delivery_days"]].mean()
    sell = sell.merge(sdur, on="seller_id")
    sell["预计天数"] = sell["estimated_days"].round(1)
    sell["实际天数"] = sell["delivery_days"].round(1)
    top_sell = sell.sort_values("超时率%", ascending=False).head(5)
    print("\n[H5 卖家] 超时率最高前5卖家（预计 vs 实际天数）：")
    print(top_sell[["seller_id", "超时率%", "count", "预计天数", "实际天数"]].to_string(index=False))
    # 承诺过紧 vs 物流延误：高超时卖家，比较其预计天数 vs 全平台中位
    med_est = sell["预计天数"].median()
    med_del = sell["实际天数"].median()
    high = sell[sell["超时率%"] >= sell["超时率%"].quantile(.9)]
    over_promise = ((high["预计天数"] < med_est) & (high["实际天数"] <= med_del)).mean()
    logistics = ((high["实际天数"] > med_del)).mean()
    print(f"\n[H5 承诺vs延误] 平台卖家预计天数中位 {med_est}、实际天数中位 {med_del}；"
          f"高手超时率卖家中：承诺过紧占 {over_promise*100:.0f}%、实际延误占 {logistics*100:.0f}%")

    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "h4h5_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: data/processed/h4h5_results.csv")

    # ============ 图1 H4 同州 vs 跨州 超时率 ============
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["同州", "跨州"], [sr*100, cr*100], color=[OK, BAD], width=.5)
    ax.bar_label(ax.containers[0], fmt="%.2f%%", fontsize=10)
    ax.set_ylabel("超时率 (%)")
    ax.set_title(f"H4 同州 vs 跨州 超时率（V={v:.3f}）")
    ax.grid(axis="y", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    ax.margins(y=.2)
    fig.tight_layout()
    fig.savefig(FIG / "h4_cross_state.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============ 图2 H5 客户州超时率 top ============
    topst = st.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(topst["customer_state"], topst["超时率%"], color=BAD)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=9)
    ax.set_xlabel("超时率 (%)")
    ax.set_title("H5 客户州超时率 top8")
    ax.grid(axis="x", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "h5_state.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============ 图3 H5 品类超时率 top ============
    topcat = cat.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(topcat["product_category_name_english"].str.replace("_", " "), topcat["超时率%"], color=BAD)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=9)
    ax.set_xlabel("超时率 (%)")
    ax.set_title("H5 品类超时率 top8")
    ax.grid(axis="x", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "h5_category.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============ 图4 H5 卖家承诺过紧 vs 物流延误 ============
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sell["预计天数"], sell["实际天数"], s=12, alpha=.4, color=C2)
    ax.axhline(med_del, color=OK, ls="--", lw=1, label=f"实际天数中位 {med_del}")
    ax.axvline(med_est, color="#C8861A", ls="--", lw=1, label=f"预计天数中位 {med_est}")
    # 承诺过紧区（左下方：预计<中位 且 实际<=中位）与物流延误区（实际>中位）
    ax.axhspan(0, med_del, xmin=0, xmax=max(0.001, (med_est-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0])),
               color="#C8861A", alpha=.08)
    ax.set_xlabel("卖家平均预计送达天数")
    ax.set_ylabel("卖家平均实际送达天数")
    ax.set_title("H5 卖家 承诺过紧（左下方） vs 物流延误（上方）")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "h5_seller.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已保存: docs/figures/h4_cross_state.png / h5_state.png / h5_category.png / h5_seller.png")


if __name__ == "__main__":
    main()
