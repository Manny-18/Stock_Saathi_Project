"""
inventory_tools.py
Shared logic for StockSaathi, the AI stock assistant for small retailers.

Everything numeric happens here, in plain Python. The chatbot (Gemini) only
understands the question, calls these functions and explains the answer, so
stock figures, reorder quantities and expiry dates can never be invented.

Author: Aakash Goswami (065061), FORE School of Management
"""
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

SAFETY_Z = 1.65            # about 95% service level for safety stock
LOOKBACK_DAYS = 28         # sales window used for the average daily rate
OVERSTOCK_DAYS = 60        # more than this many days of cover = overstock
MAX_LIST = 15              # cap list lengths returned to the chatbot


# ------------------------------------------------------------------
# 1. Loading data and computing stock metrics
# ------------------------------------------------------------------
def load_data(data_dir="data"):
    d = Path(data_dir)
    skus = pd.read_csv(d / "skus.csv", parse_dates=["earliest_expiry"])
    sales = pd.read_csv(d / "sales_daily.csv", parse_dates=["date"])
    distributors = pd.read_csv(d / "distributors.csv")
    return skus, sales, distributors


def compute_metrics(skus, sales, distributors):
    """One row per SKU with sales velocity, cover, reorder and expiry figures."""
    as_of = sales["date"].max()
    recent = sales[sales["date"] > as_of - pd.Timedelta(days=LOOKBACK_DAYS)]
    last7 = sales[sales["date"] > as_of - pd.Timedelta(days=7)]

    # pivot so days with zero sales count as zero (not missing)
    pivot = (recent.pivot_table(index="sku_id", columns="date", values="units",
                                aggfunc="sum", fill_value=0))
    avg = pivot.mean(axis=1).rename("avg_daily_sales")
    std = pivot.std(axis=1, ddof=0).rename("std_daily_sales")
    sold7 = last7.groupby("sku_id")["units"].sum().rename("sold_last_7_days")
    sold28 = recent.groupby("sku_id")["units"].sum().rename("sold_last_28_days")

    m = (skus.merge(distributors, on="distributor", how="left")
             .merge(avg, left_on="sku_id", right_index=True, how="left")
             .merge(std, left_on="sku_id", right_index=True, how="left")
             .merge(sold7, left_on="sku_id", right_index=True, how="left")
             .merge(sold28, left_on="sku_id", right_index=True, how="left"))
    for c in ["avg_daily_sales", "std_daily_sales", "sold_last_7_days", "sold_last_28_days"]:
        m[c] = m[c].fillna(0)

    lt, rp = m["lead_time_days"], m["visit_every_days"]
    m["safety_stock"] = SAFETY_Z * m["std_daily_sales"] * np.sqrt(lt)
    m["reorder_point"] = m["avg_daily_sales"] * lt + m["safety_stock"]
    m["order_up_to"] = m["avg_daily_sales"] * (lt + rp) + m["safety_stock"]
    m["days_of_cover"] = np.where(m["avg_daily_sales"] > 0,
                                  m["current_stock"] / m["avg_daily_sales"], np.inf)

    def round_to_case(row):
        need = row["order_up_to"] - row["current_stock"]
        if need <= 0 or row["avg_daily_sales"] == 0:
            return 0
        return int(math.ceil(need / row["case_size"]) * row["case_size"])

    m["suggested_order_qty"] = m.apply(round_to_case, axis=1)

    def status(row):
        if row["current_stock"] <= 0:
            return "Out of stock" if row["avg_daily_sales"] > 0 else "Not stocked"
        if row["sold_last_28_days"] == 0:
            return "No sales in 28 days"
        if row["days_of_cover"] < row["lead_time_days"]:
            return "Critical"            # will run out before a new order can arrive
        if row["current_stock"] <= row["reorder_point"]:
            return "Low"
        if row["days_of_cover"] > OVERSTOCK_DAYS:
            return "Overstock"
        return "OK"

    m["status"] = m.apply(status, axis=1)

    # expiry risk: how many units of the oldest batch will not sell in time
    m["days_to_expiry"] = (m["earliest_expiry"] - as_of).dt.days
    sellable = m["avg_daily_sales"] * m["days_to_expiry"].clip(lower=0)
    m["units_at_risk"] = (m["expiry_batch_qty"] - sellable).clip(lower=0).round().astype(int)
    m["value_at_risk_rs"] = (m["units_at_risk"] * m["cost_price"]).round(0)
    m["stock_value_rs"] = m["current_stock"] * m["cost_price"]
    m.attrs["as_of"] = as_of.strftime("%Y-%m-%d")   # string keeps attrs serialisable
    return m


# ------------------------------------------------------------------
# 2. Product matching (English, Hindi and Hinglish names)
# ------------------------------------------------------------------
STOPWORDS = {"kitna", "kitni", "kitne", "bacha", "bachi", "bache", "hai", "hain", "ka", "ki",
             "ke", "stock", "how", "much", "many", "left", "is", "are", "the", "of", "do",
             "we", "have", "in", "for", "me", "check", "show", "batao", "bata", "kya",
             "mera", "meri", "mere", "please", "pls", "abhi", "dukan", "shop", "store",
             "units", "pack", "packet", "packets", "bottle", "any", "there", "what", "about",
             "and", "aur", "status", "ko", "se", "my", "our", "products", "product",
             "items", "saman", "samaan", "wala", "wale", "brand", "all", "sab",
             "कितना", "कितनी", "कितने", "बचा", "बची", "बचे", "है", "हैं", "का", "की", "के",
             "क्या", "स्टॉक", "मेरे", "पास", "दुकान", "में", "बताओ"}


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text):
    return [t for t in normalize(text).split() if t not in STOPWORDS]


def find_products(m, query):
    """Return (match_type, DataFrame of matches).
    match_type: 'brand', 'category', 'product', 'ambiguous', 'partial' or 'none'."""
    q = normalize(query)
    qt = _tokens(query)
    if not qt:
        return "none", m.iloc[0:0]
    q_clean = " ".join(qt)

    # whole-brand or whole-category questions ("Dabur", "biscuits")
    for col, kind in (("brand", "brand"), ("category", "category")):
        vals = {normalize(v): v for v in m[col].unique()}
        if q_clean in vals:
            return kind, m[m[col] == vals[q_clean]]

    scores, vocab = [], []
    for _, row in m.iterrows():
        names = [row["product"], row["brand"] + " " + row["product"]]
        aliases = [a for a in str(row["aliases"]).split("|") if a]
        words = set(_tokens(" ".join(names + aliases)))
        vocab.append(words)
        best = 0.0
        for a in aliases:
            na = normalize(a)
            if na and (na == q_clean or re.search(rf"(^|\s){re.escape(na)}($|\s)", q)):
                best = max(best, 1.0 if na == q_clean else 0.9)
        name_tokens = set(_tokens(" ".join(names)))
        hit = [w for w in set(qt) if any(SequenceMatcher(None, w, v).ratio() >= 0.8
                                         for v in name_tokens)]
        overlap = len(hit) / len(set(qt)) if name_tokens else 0
        best = max(best, 0.85 * overlap)
        for n in names + aliases:
            best = max(best, 0.8 * SequenceMatcher(None, q_clean, normalize(n)).ratio())
        # extra words that also appear in the name (size, variant) break ties: "Maggi 70g"
        if len(set(qt)) > 1:
            best += 0.15 * overlap
        scores.append(best)

    scored = m.assign(_score=scores, _vocab=vocab).sort_values("_score", ascending=False)
    top = scored["_score"].iloc[0]
    if top < 0.55:
        return "none", scored.head(3).drop(columns="_vocab")   # nearest suggestions only
    close = scored[scored["_score"] > top - 0.04].head(6)

    # words the owner said that match none of the candidates ("Dairy Milk", "Vim liquid")
    # mean this is only a partial match: the bot must not present it as the product asked for
    def fuzzy_in(w, words):
        return any(SequenceMatcher(None, w, v).ratio() >= 0.8 for v in words)
    unmatched = [w for w in qt if not any(fuzzy_in(w, v) for v in close["_vocab"])]
    close = close.drop(columns="_vocab")
    close.attrs["unmatched_words"] = unmatched
    if unmatched:
        return "partial", close
    if len(close) == 1:
        return "product", close
    return "ambiguous", close


# ------------------------------------------------------------------
# 3. Answer builders (used by the chatbot tools AND the offline fallback)
# ------------------------------------------------------------------
def _num(x, nd=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def sku_card(row):
    cover = row["days_of_cover"]
    return {
        "product": row["product"], "brand": row["brand"],
        "current_stock": int(row["current_stock"]), "unit": row["unit"],
        "sold_last_7_days": int(row["sold_last_7_days"]),
        "avg_daily_sales": _num(row["avg_daily_sales"]),
        "days_of_cover": None if math.isinf(cover) else _num(cover),
        "status": row["status"],
        "suggested_order_qty": int(row["suggested_order_qty"]),
        "case_size": int(row["case_size"]),
        "distributor": row["distributor"], "visit_day": row["visit_day"],
        "lead_time_days": int(row["lead_time_days"]),
        "earliest_expiry": row["earliest_expiry"].strftime("%d %b %Y"),
        "mrp_rs": _num(row["mrp"], 0),
    }


def check_stock(m, product):
    kind, hits = find_products(m, product)
    if kind == "none":
        return {"found": False, "query": product,
                "message": "No product in this store matches that name.",
                "did_you_mean": hits["product"].tolist()}
    if kind == "partial":
        words = hits.attrs.get("unmatched_words", [])
        return {"found": False, "partial_match": True, "query": product,
                "unmatched_words": words,
                "message": (f"No exact match. The word(s) {words} match nothing in this store, so "
                            "the store probably does not stock that item. Closest products are "
                            "listed only as suggestions; do NOT give their stock as the answer."),
                "closest_products": [{"product": r["product"], "current_stock": int(r["current_stock"])}
                                     for _, r in hits.head(4).iterrows()]}
    if kind == "ambiguous":
        return {"found": True, "needs_clarification": True, "query": product,
                "message": "Several products match. Ask which one the owner means.",
                "options": [{"product": r["product"], "current_stock": int(r["current_stock"])}
                            for _, r in hits.iterrows()]}
    hits = hits.head(MAX_LIST)
    return {"found": True, "match_type": kind, "count": len(hits),
            "items": [sku_card(r) for _, r in hits.iterrows()]}


URGENCY = {"Out of stock": 0, "Critical": 1, "Low": 2}


def _filter(m, brand_or_category):
    f = normalize(brand_or_category or "")
    if not f or f in {"all", "sab", "sabhi", "everything"}:
        return m, None
    for col in ("brand", "category", "distributor"):
        mask = m[col].map(normalize).str.contains(f, regex=False)
        if mask.any():
            return m[mask], None
    return m.iloc[0:0], f"No brand, category or distributor called '{brand_or_category}'."


def low_stock(m, brand_or_category=""):
    sub, err = _filter(m, brand_or_category)
    if err:
        return {"error": err, "valid_brands": sorted(m["brand"].unique().tolist())}
    low = sub[sub["status"].isin(URGENCY)].copy()
    low["_u"] = low["status"].map(URGENCY)
    low = low.sort_values(["_u", "days_of_cover"])
    return {"count": len(low),
            "items": [{"product": r["product"], "status": r["status"],
                       "current_stock": int(r["current_stock"]),
                       "days_of_cover": _num(r["days_of_cover"]),
                       "suggested_order_qty": int(r["suggested_order_qty"]),
                       "distributor": r["distributor"], "visit_day": r["visit_day"]}
                      for _, r in low.head(MAX_LIST).iterrows()],
            "note": "Critical = will run out before a new order can arrive."}


def expiring_soon(m, within_days=30):
    note = None
    try:
        within_days = int(within_days)
    except (TypeError, ValueError):
        within_days, note = 30, "Days not understood, used 30."
    if within_days < 1 or within_days > 365:
        within_days = min(max(within_days, 1), 365)
        note = f"Days must be 1 to 365; used {within_days}."
    exp = m[(m["days_to_expiry"] <= within_days) & (m["expiry_batch_qty"] > 0)]
    exp = exp.sort_values("days_to_expiry")
    out = {"within_days": within_days, "count": len(exp),
           "total_value_at_risk_rs": _num(exp["value_at_risk_rs"].sum(), 0),
           "items": [{"product": r["product"], "expiry_date": r["earliest_expiry"].strftime("%d %b %Y"),
                      "days_left": int(r["days_to_expiry"]),
                      "units_in_batch": int(r["expiry_batch_qty"]),
                      "units_likely_unsold": int(r["units_at_risk"]),
                      "value_at_risk_rs": _num(r["value_at_risk_rs"], 0)}
                     for _, r in exp.head(MAX_LIST).iterrows()]}
    if note:
        out["note"] = note
    return out


def purchase_order(m, distributor_or_brand=""):
    sub, err = _filter(m, distributor_or_brand)
    if err:
        return {"error": err, "valid_distributors": sorted(m["distributor"].unique().tolist())}
    po = sub[(sub["suggested_order_qty"] > 0) & (sub["status"].isin(URGENCY))]
    groups = []
    for dist, g in po.groupby("distributor"):
        lines = [{"product": r["product"], "order_qty": int(r["suggested_order_qty"]),
                  "case_size": int(r["case_size"]),
                  "est_cost_rs": _num(r["suggested_order_qty"] * r["cost_price"], 0)}
                 for _, r in g.sort_values("product").iterrows()]
        groups.append({"distributor": dist, "visit_day": g["visit_day"].iloc[0],
                       "lines": lines, "total_rs": _num(sum(l["est_cost_rs"] for l in lines), 0)})
    return {"orders": groups, "grand_total_rs": _num(sum(g["total_rs"] for g in groups), 0),
            "note": "Draft only. Quantities top stock up to cover lead time + time to next visit "
                    "+ safety stock, rounded up to full cases."}


def sales_report(m, sales, period_days=7, brand_or_category=""):
    note = None
    try:
        period_days = int(period_days)
    except (TypeError, ValueError):
        period_days, note = 7, "Period not understood, used 7 days."
    if not 1 <= period_days <= 90:
        period_days = min(max(period_days, 1), 90)
        note = f"Period must be 1 to 90 days; used {period_days}."
    sub, err = _filter(m, brand_or_category)
    if err:
        return {"error": err}
    as_of = pd.Timestamp(m.attrs["as_of"])
    s = sales[(sales["date"] > as_of - pd.Timedelta(days=period_days))
              & (sales["sku_id"].isin(sub["sku_id"]))]
    agg = (s.groupby("sku_id")["units"].sum().reindex(sub["sku_id"], fill_value=0)
             .rename("units").reset_index()
             .merge(sub[["sku_id", "product", "mrp"]], on="sku_id"))
    agg["revenue_rs"] = agg["units"] * agg["mrp"]
    top = agg.sort_values("revenue_rs", ascending=False).head(5)
    slow = agg.sort_values("units").head(5)
    out = {"period_days": period_days, "total_units": int(agg["units"].sum()),
           "total_revenue_rs": _num(agg["revenue_rs"].sum(), 0),
           "top_sellers_by_revenue": [{"product": r["product"], "units": int(r["units"]),
                                       "revenue_rs": _num(r["revenue_rs"], 0)} for _, r in top.iterrows()],
           "slowest_movers": [{"product": r["product"], "units": int(r["units"])} for _, r in slow.iterrows()]}
    if note:
        out["note"] = note
    return out


def dead_stock(m):
    d = m[m["status"].isin(["No sales in 28 days", "Overstock"])].sort_values(
        "stock_value_rs", ascending=False)
    return {"count": len(d), "value_locked_rs": _num(d["stock_value_rs"].sum(), 0),
            "items": [{"product": r["product"], "status": r["status"],
                       "current_stock": int(r["current_stock"]),
                       "sold_last_28_days": int(r["sold_last_28_days"]),
                       "stock_value_rs": _num(r["stock_value_rs"], 0)}
                      for _, r in d.head(MAX_LIST).iterrows()]}


# ------------------------------------------------------------------
# 4. Offline fallback: keyword routing when the AI is unavailable
# ------------------------------------------------------------------
INTENT_WORDS = {
    "low": ["low", "khatam", "kam ", "reorder", "running out", "finish", "short", "mangana",
            "मंगाना", "खत्म", "कम"],
    "expiry": ["expir", "expiry", "kharab", "date", "एक्सपायरी", "खराब"],
    "order": ["order", "po ", "purchase", "aadesh", "ऑर्डर"],
    "sales": ["sale", "sold", "bika", "bik ", "best", "top", "selling", "बिक"],
    "dead": ["dead", "nahi bik", "not selling", "overstock", "zyada"],
}


def offline_answer(m, sales, text):
    """Rule-based answer used only when Gemini is down. Returns markdown."""
    t = " " + normalize(text) + " "
    def has(k):
        return any(w in t for w in INTENT_WORDS[k])

    if has("dead"):
        r = dead_stock(m)
        lines = [f"- {i['product']}: {i['current_stock']} in stock, {i['sold_last_28_days']} sold in 28 days"
                 for i in r["items"][:8]]
        return f"**Slow or dead stock** (Rs {r['value_locked_rs']:,.0f} locked):\n" + "\n".join(lines)
    if has("expiry"):
        r = expiring_soon(m, 30)
        lines = [f"- {i['product']}: expires {i['expiry_date']} ({i['days_left']} day{'s' if i['days_left'] != 1 else ''}), "
                 f"{i['units_likely_unsold']} units likely unsold" for i in r["items"][:8]]
        return "**Expiring in the next 30 days:**\n" + ("\n".join(lines) or "Nothing.")
    if has("order") and not has("low"):
        r = purchase_order(m, "")
        parts = [f"**{o['distributor']}** ({o['visit_day']}): " +
                 ", ".join(f"{l['product']} x{l['order_qty']}" for l in o["lines"]) for o in r["orders"]]
        return "**Draft orders:**\n\n" + "\n\n".join(parts)
    if has("low"):
        r = low_stock(m, "")
        lines = [f"- {i['product']}: {i['status']}, {i['current_stock']} left, order {i['suggested_order_qty']}"
                 for i in r["items"][:10]]
        return "**Items to reorder:**\n" + "\n".join(lines)
    if has("sales"):
        r = sales_report(m, sales, 7, "")
        lines = [f"- {i['product']}: {i['units']} units, Rs {i['revenue_rs']:,.0f}"
                 for i in r["top_sellers_by_revenue"]]
        return "**Top sellers, last 7 days:**\n" + "\n".join(lines)
    r = check_stock(m, text)
    if r.get("partial_match"):
        return ("No exact match for that. Closest items in stock: " +
                ", ".join(o["product"] for o in r["closest_products"]) + ".")
    if not r["found"]:
        return ("I could not match that to a product. Try a product name like 'Maggi' or "
                "'Parle-G', or open the Store dashboard tab.")
    if r.get("needs_clarification"):
        return "Several products match: " + ", ".join(o["product"] for o in r["options"]) + \
               ". Please type the full name."
    lines = [f"- **{i['product']}**: {i['current_stock']} {i['unit']} left, {i['sold_last_7_days']} sold "
             f"in 7 days, status {i['status']}" for i in r["items"]]
    return "\n".join(lines)
