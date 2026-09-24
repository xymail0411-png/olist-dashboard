# -*- coding: utf-8 -*-
"""
07 H6 主假设验证：体验 -> 满意 -> 复购（结构 vs 体验之争的胜负手）
数据源：
  - data/processed/orders_master.csv        订单主宽表（is_late 超时 / review_score 评分 / 时间）
  - data/processed/customer_rfm.csv         两阶段分层结果（stage: 单次购买/复购客户）
  - data/olist/olist_order_reviews_dataset.csv 原始评语（低分文本挖掘）
产出：
  - data/processed/h6_results.csv           三段检验结果汇总
  - docs/figures/h6_late_vs_on_time.png     ① 体验->满意：超时 vs 准时 评分对比
  - docs/figures/h6_score_repeat.png        ② 满意->复购：评分档 x 复购率
  - docs/figures/h6_lowreview_words.png     ③ 低分评语词频

H6 表述：超时订单评分显著低于准时；高评分客户复购率更高。
三段链：体验(是否超时) -> 满意(评分) -> 复购(是否再次购买)

设计要点（重要）：
  1) 大样本下 p 值几乎必然显著，结论以效应量为准（与 H1 口径一致）
  2) 订单级 vs 客户级分开：
      ① 体验->满意 用「每笔订单」的 is_late x review_score
      ② 满意->复购 用「每个客户」的首单评分，并 merge stage（单次/复购）
  3) 用「首单」而非「平均」：复购客有 2+ 单，平均分被多单稀释，
     不能归因"体验好才复购"；首单评分才干净地预测后续是否复购。

用法：
    C:\\...\\python.exe scripts/07_h6_test.py
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

C1, C2 = "#C9D0D9", "#52647D"   # 准时(浅) / 超时(深)
OK, BAD = "#52647D", "#B84C3A"  # 正向(藏青) / 负向(暖赤)


def mw_test(a, b):
    """Mann-Whitney U 双侧检验，返回 (U, p, 秩双列相关 r)。"""
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = 1 - 2 * u / (n1 * n2)
    return u, p, r


def eff_label_r(r):
    ar = abs(r)
    return "大" if ar > .5 else "中" if ar > .3 else "小" if ar > .1 else "无"


def eff_label_v(v):
    return "大" if v > .3 else "中" if v > .1 else "小"


def cramers_v(chi2, n, r, k):
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def main():
    master = pd.read_csv(PROCESSED / "orders_master.csv",
                         parse_dates=["order_purchase_timestamp"]).sort_values("order_purchase_timestamp")
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")

    rows = []

    # ============ ① 体验 -> 满意（订单级）============
    scored = master[master["review_score"] > 0].copy()
    late = scored[scored["is_late"] == 1]["review_score"]
    ontime = scored[scored["is_late"] == 0]["review_score"]
    u, p, r = mw_test(ontime, late)          # r>0 表示准时组评分更高
    print(f"[① 体验->满意] 准时评分 {ontime.mean():.2f} (n={len(ontime):,}) vs 超时 {late.mean():.2f} (n={len(late):,}) | r={r:.3f} ({eff_label_r(r)}效应)")
    rows.append({"段": "①体验->满意", "对比": "准时 vs 超时 评分",
                 "准时均值": round(ontime.mean(), 3), "超时均值": round(late.mean(), 3),
                 "检验": "Mann-Whitney U", "p值": p, "效应量r": r,
                 "结论": "超时评分显著更低" if r < 0 else "超时评分无差异"})

    # ============ ② 满意 -> 复购（客户级，用首单）============
    # 首单：按客户取最早一笔
    first = master.sort_values("order_purchase_timestamp").groupby("customer_unique_id").head(1)
    f = first[["customer_unique_id", "review_score", "is_late"]].rename(
        columns={"review_score": "首单评分", "is_late": "首单超时"})
    cust = f.merge(rfm[["customer_unique_id", "stage"]], on="customer_unique_id", how="left")
    cust["复购"] = (cust["stage"] == "复购客户").astype(int)

    # 评分档 -> 复购率
    def score_band(s):
        return "1-2低分" if s <= 2 else "3中分" if s == 3 else "4-5高分"
    cust = cust[cust["首单评分"] > 0].copy()
    cust["评分档"] = cust["首单评分"].map(score_band)
    band_rate = cust.groupby("评分档")["复购"].agg(["mean", "count"]).reindex(["1-2低分", "3中分", "4-5高分"])
    band_rate["复购率%"] = (band_rate["mean"] * 100).round(2)
    # 卡方：评分档 x 是否复购
    ct = pd.crosstab(cust["评分档"], cust["复购"])
    chi2, p2, dof, _ = stats.chi2_contingency(ct.values)
    v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
    print("\n[② 满意->复购] 评分档 x 复购率：")
    print(band_rate["复购率%"].to_string())
    print(f"  卡方 p={p2:.2e} | Cramér's V={v:.3f} ({eff_label_v(v)}效应)")
    rows.append({"段": "②满意->复购", "对比": "评分档 x 复购率",
                 "低分复购率%": round(band_rate.loc["1-2低分", "复购率%"], 2),
                 "中分复购率%": round(band_rate.loc["3中分", "复购率%"], 2),
                 "高分复购率%": round(band_rate.loc["4-5高分", "复购率%"], 2),
                 "检验": "卡方", "p值": p2, "效应量V": v,
                 "结论": "评分越高复购率越高" if v > .1 else "评分与复购关系弱"})

    # ============ ③ 体验 -> 复购（首单是否超时 x 复购率）============
    late_rate = cust[cust["首单超时"] == 1]["复购"].mean()
    ontime_rate = cust[cust["首单超时"] == 0]["复购"].mean()
    ct2 = pd.crosstab(cust["首单超时"], cust["复购"])
    chi2b, pb, _, _ = stats.chi2_contingency(ct2.values)
    vb = cramers_v(chi2b, ct2.values.sum(), ct2.shape[0], ct2.shape[1])
    print(f"\n[③ 体验->复购] 首单准时客户复购率 {ontime_rate*100:.2f}% vs 首单超时 {late_rate*100:.2f}% | V={vb:.3f}")
    rows.append({"段": "③体验->复购", "对比": "首单是否超时 x 复购率",
                 "准时复购率%": round(ontime_rate * 100, 2), "超时复购率%": round(late_rate * 100, 2),
                 "检验": "卡方", "p值": pb, "效应量V": vb,
                 "结论": "首单超时显著降低复购率" if vb > .1 and late_rate < ontime_rate else "首单超时对复购影响弱"})

    # ============ ④ 低分评语文本挖掘（定性）============
    reviews = pd.read_csv(RAW / "olist_order_reviews_dataset.csv")
    low = reviews[(reviews["review_score"] <= 2) & reviews["review_comment_message"].notna()]
    text = " ".join(low["review_comment_message"].astype(str))
    # 简单词频：去葡语虚词/助词（无差评信息量），保留业务主题词（pedido/produto/entrega/prazo 等）
    stop = set("""a o e de do da dos das em no na nos nas para por com sem que não nao
                   muito mais menos foi foram ser ter tem ao aos à às meu minha seus sua
                   me te se lhe nos vos lhes eu tu ele ela nós vós eles elas você vocês
                   está estao estão estou estamos como quero apenas pois dois agora ainda
                   já tambem também tudo mesmo quando onde sempre nada algo cada esta este
                   isso isto aqui ali bem depois antes então assim porque outro outros vez
                   qual quais pode podem fazer feito tenho vamos vai vou sou são eram havia
                   tinha essa esse é ou se um uma uns umas os as oi ok""".split())
    words = [w.lower() for w in text.split() if w.isalpha() and len(w) >= 4 and w.lower() not in stop]
    from collections import Counter
    top = Counter(words).most_common(15)
    print("\n[④ 低分评语] 1-2星评语高频词（前15）：")
    for w, c in top:
        print(f"  {w}: {c}")

    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "h6_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: data/processed/h6_results.csv")

    # ============ 图1 体验->满意：超时 vs 准时 评分分布 ============
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.arange(0.5, 6, 1)
    ax.hist([ontime, late], bins=bins, density=True, histtype="bar",
            color=[C1, C2], label=["准时", "超时"], alpha=.85)
    ax.axvline(ontime.mean(), color=OK, ls="--", lw=1.2)
    ax.axvline(late.mean(), color=BAD, ls="--", lw=1.2)
    ax.text(ontime.mean() + .05, ax.get_ylim()[1] * .9, f"准时 {ontime.mean():.2f}", color=OK, fontsize=9)
    ax.text(late.mean() + .05, ax.get_ylim()[1] * .7, f"超时 {late.mean():.2f}", color=BAD, fontsize=9)
    ax.set_xlabel("评分 (1-5)")
    ax.set_ylabel("密度")
    ax.set_title("H6 ①体验->满意：准时 vs 超时 订单评分分布")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "h6_late_vs_on_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============ 图2 满意->复购：评分档 x 复购率 ============
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = [BAD, "#C8861A", OK]   # 低分暖赤 / 中分琥珀 / 高分藏青
    ax.bar(band_rate.index, band_rate["复购率%"], color=colors, width=.55)
    ax.bar_label(ax.containers[0], fmt="%.2f%%", fontsize=10)
    ax.set_ylabel("复购率 (%)")
    ax.set_title("H6 ②满意->复购：首单评分档 x 复购率")
    ax.grid(axis="y", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    ax.margins(y=.2)
    fig.tight_layout()
    fig.savefig(FIG / "h6_score_repeat.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ============ 图3 低分评语词频 ============
    words = [w for w, _ in top][::-1]
    counts = [c for _, c in top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(words, counts, color=BAD)
    ax.bar_label(ax.containers[0], fontsize=9)
    ax.set_xlabel("出现次数")
    ax.set_title("H6 ③低分评语高频词（1-2星，前15）")
    ax.grid(axis="x", color="#E2E2E2", linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "h6_lowreview_words.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已保存: docs/figures/h6_late_vs_on_time.png / h6_score_repeat.png / h6_lowreview_words.png")


if __name__ == "__main__":
    main()
