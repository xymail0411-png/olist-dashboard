# -*- coding: utf-8 -*-
"""
10b 时间窗复购预测（补充对照）：
  全期预测（10 号脚本）label = 最终是否复购（正样本 3%，AUC≈0.59，弱）
  本脚本改为「首购后 30 天内是否再购」，并做右侧截尾控制，与全期两相对照，
  说明：预测难是任务太稀疏，还是方法不行？换到可操作短期窗口，留旧是否有抓手？
特征 = 首单特征（金额/评分/超时/配送/承诺差/品类/州/月份）——首购时刻即可得，无泄漏
产出：data/processed/repeat_window_results.csv、repeat_window_topk.csv
      docs/figures/window_auc_compare.png、window_topk.png、window_feature_importance.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_features():
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")
    master = pd.read_csv(PROCESSED / "orders_master.csv")
    master = master.sort_values("order_purchase_timestamp")
    master["ts"] = pd.to_datetime(master["order_purchase_timestamp"])
    master["order_rank"] = master.groupby("customer_unique_id").cumcount() + 1

    first = master[master["order_rank"] == 1].copy()
    second = master[master["order_rank"] == 2][["customer_unique_id", "ts"]].rename(
        columns={"ts": "second_ts"})

    first["first_month"] = first["ts"].dt.month
    first["promise_gap"] = first["estimated_days"] - first["delivery_days"]  # 承诺天数-实际配送=配送快慢

    agg = first.groupby("customer_unique_id").agg(
        first_ts=("ts", "first"),
        first_value=("total_value", "first"),
        first_score=("review_score", lambda s: s.iloc[0]),
        first_late=("is_late", lambda s: float(s.iloc[0])),
        first_delivery_days=("delivery_days", "first"),
        promise_gap=("promise_gap", "first"),
        first_month=("first_month", "first"),
        state=("customer_state", "first"),
    ).reset_index()
    agg["first_late"] = agg["first_late"].astype(float)

    items = pd.read_csv(ROOT / "data" / "olist" / "olist_order_items_dataset.csv")
    prods = pd.read_csv(PROCESSED / "products_clean.csv")[["product_id", "product_category_name_english"]]
    ci = items.merge(prods, on="product_id", how="left").merge(
        first[["order_id", "customer_unique_id"]], on="order_id", how="left")
    ci["cat"] = ci["product_category_name_english"].fillna("unknown")
    fcat = ci.groupby("customer_unique_id")["cat"].agg(lambda s: s.mode().iloc[0]).rename("first_category")
    agg = agg.merge(fcat, on="customer_unique_id", how="left")

    df = rfm[["customer_unique_id", "stage"]].merge(agg, on="customer_unique_id", how="left")

    # 两个标签
    df["label_full"] = (df["stage"] == "复购客户").astype(int)
    df = df.merge(second, on="customer_unique_id", how="left")
    gap_days = (df["second_ts"] - df["first_ts"]).dt.days
    df["label_30"] = ((gap_days <= 30) & gap_days.notna()).astype(int)

    # 右侧截尾控制：只保留「首购距数据末尾 >= 30 天」的客户做 30 天评估
    end_ts = master["ts"].max()
    df["window_ok"] = (end_ts - df["first_ts"]).dt.days >= 30

    df["state_id"] = df["state"].fillna("SP").astype("category").cat.codes
    df["first_category_id"] = df["first_category"].fillna("unknown").astype("category").cat.codes
    feat_cols = ["first_value", "first_score", "first_late", "first_delivery_days",
                 "promise_gap", "first_month", "state_id", "first_category_id"]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    return df, feat_cols


def run_task(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.fit_transform(X_tr), y_tr)
    lr_prob = lr.predict_proba(sc.transform(X_te))[:, 1]
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbose=-1).fit(X_tr, y_tr)
    gbm_prob = gbm.predict_proba(X_te)[:, 1]
    return lr_prob, gbm_prob, lr, gbm


def topk_hit(prob, y_true, k):
    cutoff = int(len(prob) * k / 100)
    if cutoff == 0:
        return 0, 0
    top = np.argsort(-prob)[:cutoff]
    n_hit = int(y_true.iloc[top].sum())
    return n_hit, n_hit / int(y_true.sum())


def main():
    df, feat_cols = build_features()
    X = df[feat_cols].copy()

    # ===== 任务A：全期复购（对照，同 10） =====
    yA = df["label_full"]
    # ===== 任务B：30 天复购（只留窗口完整客户） =====
    maskB = df["window_ok"]
    XB = X[maskB]; yB = df.loc[maskB, "label_30"]
    print(f"全期: 客户 {len(df):,}，正样本 {yA.sum():,}({yA.mean()*100:.2f}%)")
    print(f"30天窗(截尾后): 客户 {len(XB):,}，正样本 {yB.sum():,}({yB.mean()*100:.2f}%)")

    rows = []
    # 任务A
    X_tr, X_te, yA_tr, yA_te = train_test_split(X, yA, test_size=0.3, stratify=yA, random_state=42)
    lrp, gbp, lrA, gbmA = run_task(X_tr, yA_tr, X_te, yA_te)
    rows.append(["全期复购", "逻辑回归", roc_auc_score(yA_te, lrp), average_precision_score(yA_te, lrp), int(yA_te.sum())])
    rows.append(["全期复购", "LightGBM", roc_auc_score(yA_te, gbp), average_precision_score(yA_te, gbp), int(yA_te.sum())])
    # 任务B
    XB_tr, XB_te, yB_tr, yB_te = train_test_split(XB, yB, test_size=0.3, stratify=yB, random_state=42)
    lrp2, gbp2, lrB, gbmB = run_task(XB_tr, yB_tr, XB_te, yB_te)
    rows.append(["30天复购", "逻辑回归", roc_auc_score(yB_te, lrp2), average_precision_score(yB_te, lrp2), int(yB_te.sum())])
    rows.append(["30天复购", "LightGBM", roc_auc_score(yB_te, gbp2), average_precision_score(yB_te, gbp2), int(yB_te.sum())])
    res = pd.DataFrame(rows, columns=["任务", "模型", "AUC", "PR_AUC", "正样本"])
    res.to_csv(PROCESSED / "repeat_window_results.csv", index=False, encoding="utf-8-sig")
    print("\n对照结果：")
    print(res.to_string(index=False))

    # 30 天 Top-K（预算视角）
    tk = []
    for k in [1, 2, 3, 5, 10]:
        lh, lcov = topk_hit(lrp2, yB_te, k)
        gh, gcov = topk_hit(gbp2, yB_te, k)
        tk.append([k, lh, lcov*100, gh, gcov*100])
    tkdf = pd.DataFrame(tk, columns=["top_k%", "LR命中", "LR覆盖率%", "GBM命中", "GBM覆盖率%"])
    tkdf[["LR覆盖率%", "GBM覆盖率%"]] = tkdf[["LR覆盖率%", "GBM覆盖率%"]].round(1)
    tkdf.to_csv(PROCESSED / "repeat_window_topk.csv", index=False, encoding="utf-8-sig")
    print("\n30 天窗 Top-K 命中：")
    print(tkdf.to_string(index=False))

    # ===== 图1：AUC/PR-AUC 对照（全期 vs 30天 × LR/GBM） =====
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tasks = ["全期复购", "30天复购"]
    for i, t in enumerate(tasks):
        sub = res[res["任务"] == t]
        for j, m in enumerate(["逻辑回归", "LightGBM"]):
            v = sub[sub["模型"] == m]["AUC"].iloc[0]
            xp = i + (j - 0.5) * 0.3
            ax.bar(xp, v, 0.3, color="#52647D" if m == "LightGBM" else "#C9D0D9",
                   label=m if i == 0 else None)
            ax.text(xp, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#B84C3A", linestyle="--", linewidth=1)
    ax.text(1.95, 0.51, "随机 0.5", color="#B84C3A", fontsize=8, ha="right")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks)
    ax.set_ylabel("AUC")
    ax.set_title("复购预测 AUC 对照：全期 vs 首购后 30 天")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#E2E2E2", linewidth=.5); ax.set_axisbelow(True); ax.margins(y=.2)
    fig.tight_layout(); fig.savefig(FIG / "window_auc_compare.png", dpi=130); plt.close(fig)

    # ===== 图2：30 天 Top-K =====
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = tkdf["top_k%"]
    ax.plot(x, tkdf["LR覆盖率%"], marker="o", label="逻辑回归", color="#8D9AAB")
    ax.plot(x, tkdf["GBM覆盖率%"], marker="s", label="LightGBM", color="#52647D")
    for xi, yi in zip(x, tkdf["LR覆盖率%"]):
        ax.text(xi, yi + 1, f"{yi:.0f}", ha="center", fontsize=8)
    for xi, yi in zip(x, tkdf["GBM覆盖率%"]):
        ax.text(xi, yi - 2.5, f"{yi:.0f}", ha="center", fontsize=8, color="#52647D")
    ax.set_xlabel("取概率最高前 k% 客户"); ax.set_ylabel("捕获复购覆盖率 %")
    ax.set_title("30 天窗复购 Top-K 命中（预算视角）")
    ax.legend(frameon=False); ax.grid(color="#E2E2E2", linewidth=.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "window_topk.png", dpi=130); plt.close(fig)

    # ===== 图3：30 天特征重要性 =====
    imp = pd.DataFrame({"imp": gbmB.feature_importances_}, index=gbmB.feature_name_).sort_values(
        "imp", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index[::-1], imp["imp"][::-1], color="#52647D")
    for yi, v in enumerate(imp["imp"][::-1]):
        ax.text(v + 2, yi, str(int(v)), va="center", fontsize=8)
    ax.set_xlabel("重要性（分裂增益）"); ax.set_title("30 天复购 LightGBM 特征重要性 Top10")
    ax.margins(x=.15)
    fig.tight_layout(); fig.savefig(FIG / "window_feature_importance.png", dpi=130); plt.close(fig)

    print("\n已保存: repeat_window_results/topk.csv + window_auc_compare/topk/feature_importance.png")


if __name__ == "__main__":
    main()
