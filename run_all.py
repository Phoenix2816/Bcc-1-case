import os
import re
import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime


# =====================
# CONFIG
# =====================
CONFIG = {
    "BASE_DIR": r"C:\Users\Admin\Desktop\Hk 1 case\case 1",
    "OUTPUT_FILE": "output.csv",
    "ENCODINGS": ["utf-8-sig", "utf-8", "cp1251"],
    "PRODUCTS": [
        "Карта для путешествий",
        "Премиальная карта",
        "Кредитная карта",
        "Обмен валют / FX",
        "Кредит наличными",
        "Вклад сберегательный (заморозка)",
        "Вклад накопительный",
        "Инвестиции (брокерский счёт)",
        "Золотые слитки",
    ],
}


# =====================
# HELPERS
# =====================
def clean_csv_header(path: str) -> None:
    """
    Fix CSV header if merged with filename or corrupted.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        return
    first_line = lines[0]
    if "client_code" in first_line.lower() and not first_line.strip().lower().startswith("client_code"):
        idx = first_line.lower().find("client_code")
        lines[0] = first_line[idx:]
    elif ".csv" in first_line and not first_line.lower().startswith("client_code"):
        parts = first_line.split(".csv", 1)
        lines[0] = parts[-1].lstrip(", \t")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def safe_read_csv(path: str, encodings: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Read CSV safely trying multiple encodings and delimiter fixes.
    """
    if encodings is None:
        encodings = CONFIG["ENCODINGS"]
    if not os.path.exists(path):
        return None
    clean_csv_header(path)
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.shape[1] == 1:  # try separator fix
                df_try = pd.read_csv(path, encoding=enc, sep=";")
                if df_try.shape[1] > 1:
                    df = df_try
            df.columns = [col.strip() for col in df.columns]
            return df
        except Exception:
            continue
    return None


def fmt_amount(val: float) -> str:
    """
    Format number with space separator, comma decimal and appended currency.
    """
    if not val:
        return "0 ₸"
    val_rounded = round(val, 2)
    s = f"{val_rounded:,.2f}".replace(",", "x").replace(".", ",").replace("x", " ")
    if s.endswith(",00"):
        s = s[:-3]
    return f"{s} ₸"


def to_num(df: pd.DataFrame, col: str = "amount") -> pd.DataFrame:
    """
    Safely convert column to numeric, filling errors with zero.
    """
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def month_name_ru(dt: datetime) -> str:
    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря"
    ]
    return months[dt.month - 1] if dt else ""


def extract_month_full_str(dates):
    if len(dates) == 0:
        return ""
    dates_parsed = pd.to_datetime(dates, errors='coerce').dropna()
    if not dates_parsed.empty:
        dt_sample = dates_parsed.mode()[0]
        day = dt_sample.day
        month = month_name_ru(dt_sample)
        year = dt_sample.year
        return f"{day} {month} {year}"
    return ""


def months_diff(start_date, end_date) -> float:
    if pd.isna(start_date) or pd.isna(end_date):
        return 3.0  # default to 3 months if unknown
    return max(1.0, (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1)


def normalize_per_month(df: pd.DataFrame, amount_col: str = "amount", date_col: str = "date") -> float:
    """
    Normalize total 'amount' over number of months spanned by 'date' column.
    """
    if df is None or df.empty or amount_col not in df.columns or date_col not in df.columns:
        return 0.0
    df_clean = df.dropna(subset=[date_col, amount_col])
    if df_clean.empty:
        return 0.0
    dates = pd.to_datetime(df_clean[date_col], errors='coerce').dropna()
    if dates.empty:
        return df_clean[amount_col].sum() / 3  # default 3 months
    months = max(1, (dates.max().year - dates.min().year) * 12 + (dates.max().month - dates.min().month) + 1)
    return df_clean[amount_col].sum() / months


def calc_travel_card(df: pd.DataFrame) -> float:
    """
    Calculate monthly cashback benefit for Travel card.
    Returns 0 if 'category' column missing or no relevant data.
    """
    if df is None or df.empty or "category" not in df.columns:
        return 0.0
    cats = ["Путешествия", "Отели", "Такси"]
    travel_df = df[df['category'].isin(cats)]
    if travel_df.empty:
        return 0.0
    # Safely handle date column missing
    if "date" not in travel_df.columns:
        return 0.0
    travel_df = travel_df.copy()
    travel_df['date'] = pd.to_datetime(travel_df['date'], errors='coerce').dt.date
    trip_days = travel_df['date'].nunique()
    monthly_sum = normalize_per_month(travel_df)
    cashback = monthly_sum * 0.04
    return cashback



def safe_filter(df: pd.DataFrame, column: str, values: List[str]) -> pd.DataFrame:
    """Return filtered dataframe by column isin values or empty if column missing."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame()
    return df[df[column].isin(values)]


def calc_premium_card(df: pd.DataFrame, avg_balance: float, transfers_df: pd.DataFrame, status: str) -> float:
    """
    Calculate monthly cashback for Premium card including saved fees,
    safely checking existence of 'category' column.
    """
    if df is None or df.empty or "category" not in df.columns:
        base_expenses = 0.0
        bonus_expenses = 0.0
    else:
        base_expenses = normalize_per_month(df)
        bonus_cats = ["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"]
        bonus_expenses = normalize_per_month(safe_filter(df, "category", bonus_cats))

    if avg_balance >= 6_000_000:
        tier = 0.04
    elif avg_balance >= 1_000_000:
        tier = 0.03
    else:
        tier = 0.02

    cashback = base_expenses * tier + bonus_expenses * 0.04
    cashback = min(cashback, 100_000)

    saved_fees = 0
    if transfers_df is not None and "type" in transfers_df.columns:
        atm_withdrawals = transfers_df[transfers_df["type"] == "atm_withdrawal"]
        saved_fees += atm_withdrawals.shape[0] * 200
        transfers_out = transfers_df[transfers_df["type"].isin(["card_out", "p2p_out"])]
        saved_fees += transfers_out.shape[0] * 100

    if status == "Премиальный клиент":
        cashback *= 1.1  # 10% premium client bonus

    return cashback + saved_fees

def calc_credit_card(df, transfers):
    if df is None or df.empty:
        return 0, []
    monthly_data = df.copy()
    monthly_data['date'] = pd.to_datetime(monthly_data['date'], errors='coerce')
    months_count = months_diff(monthly_data['date'].min(), monthly_data['date'].max())
    cat_sum = monthly_data.groupby("category")["amount"].sum() / months_count
    top3_cats = cat_sum.nlargest(3)
    top3_sum = top3_cats.sum() if not top3_cats.empty else 0
    online_cats = ["Играем дома", "Едим дома", "Смотрим дома"]
    online_sum = normalize_per_month(df[df["category"].isin(online_cats)])
    cashback = 0.10 * (top3_sum + online_sum)
    bonus = 0
    if transfers is not None and "type" in transfers.columns:
        has_installment = transfers["type"].astype(str).str.contains("installment_payment_out|cc_repayment_out", case=False).any()
        if has_installment:
            bonus = 2000
    return cashback + bonus, top3_cats.index.tolist()


def calc_fx(transfers, df):
    if transfers is None or transfers.empty:
        return 0, []
    vol = 0
    fx_currencies = set()
    if "type" in transfers.columns and "amount" in transfers.columns:
        fx_ops = transfers[transfers["type"].isin(["fx_buy", "fx_sell"])]
        vol += fx_ops["amount"].sum()
        if not fx_ops.empty and "currency" in fx_ops.columns:
            fx_currencies.update(fx_ops["currency"].dropna().unique())
    if df is not None and "currency" in df.columns:
        vol += df[df["currency"].isin(["USD", "EUR"])]["amount"].sum()
        fx_currencies.update(df[df["currency"].isin(["USD", "EUR"])]["currency"].dropna().unique())
    vol_monthly = vol / 3 if vol > 0 else 0
    return 0.02 * vol_monthly, sorted(fx_currencies)


def calc_deposit_saver(avg_balance, df):
    if df is None or df.empty or "amount" not in df.columns:
        spend = 0
        volatility = 0
    else:
        spend = normalize_per_month(df)
        volatility = df["amount"].std() if len(df["amount"]) > 1 else 0
    if avg_balance >= 500_000 and volatility < avg_balance * 0.05:
        return avg_balance * 0.165
    return 0


def calc_deposit_accum(avg_balance, transfers):
    if transfers is None or transfers.empty:
        return 0
    top_up = 0
    if "type" in transfers.columns and "amount" in transfers.columns:
        top_up = transfers[transfers["type"].str.contains("deposit_topup", case=False, na=False)]["amount"].sum()
    if avg_balance < 200_000 and top_up > 50_000:
        return avg_balance * 0.155
    return 0


def calc_credit_cash(avg_balance, transfers):
    if transfers is None or transfers.empty:
        return 0
    inflows = transfers[transfers["direction"] == "in"]["amount"].sum()
    outflows = transfers[transfers["direction"] == "out"]["amount"].sum()
    loan_payments = transfers[transfers["type"] == "loan_payment_out"]["amount"].sum()
    if avg_balance < 100_000 and outflows > inflows * 1.3 and loan_payments > 0:
        return 3000
    return 0


def calc_investments(avg_balance, transfers):
    if avg_balance <= 100_000:
        return 0
    invest_ops = 0
    if transfers is not None and "type" in transfers.columns:
        invest_ops = transfers[transfers["type"].str.contains("invest", case=False, na=False)]["amount"].sum()
    return max(500, invest_ops * 0.05)


def calc_gold(transfers):
    if transfers is None or transfers.empty:
        return 0
    gold_ops = transfers[transfers["type"].str.contains("gold", case=False, na=False)]["amount"].sum()
    if gold_ops > 100_000:
        return gold_ops * 0.01
    return 0

# Step 1: Add product to CONFIG
CONFIG["PRODUCTS"].append("Мультивалютный депозит")

# Step 2: Implement calculation function
def calc_multicurrency_deposit(transfers: Optional[pd.DataFrame]) -> float:
    """
    Calculate benefit for multi-currency deposit.
    For example, sum all foreign currency balances or amounts and apply a small interest rate reward.
    """
    if transfers is None or transfers.empty:
        return 0
    # Consider balances or amounts in foreign currencies other than KZT
    if "currency" not in transfers.columns or "amount" not in transfers.columns:
        return 0
    foreign_currencies = transfers[transfers["currency"].isin(["USD", "EUR", "RUB", "GBP", "CNY"])]
    total_foreign_amount = foreign_currencies["amount"].sum()
    # Example: 1.2% annual interest approximation divided by 12 months
    monthly_benefit = total_foreign_amount * 0.012 / 12
    return monthly_benefit

# Step 3: Update score_all
def score_all(tx_df, tr_df, avg_balance, status):
    scores = {
        "Карта для путешествий": calc_travel_card(tx_df),
        "Премиальная карта": calc_premium_card(tx_df, avg_balance, tr_df, status),
        "Кредитная карта": 0,
        "Обмен валют / FX": 0,
        "Кредит наличными": calc_credit_cash(avg_balance, tr_df),
        "Вклад сберегательный (заморозка)": calc_deposit_saver(avg_balance, tx_df),
        "Вклад накопительный": calc_deposit_accum(avg_balance, tr_df),
        "Инвестиции (брокерский счёт)": calc_investments(avg_balance, tr_df),
        "Золотые слитки": calc_gold(tr_df),
        "Мультивалютный депозит": calc_multicurrency_deposit(tr_df),
    }
    credit_cashback, credit_top3 = calc_credit_card(tx_df, tr_df)
    scores["Кредитная карта"] = credit_cashback
    return scores, credit_top3

# Step 4: Add message generation in generate_push
def generate_push(name, product, df, benefit, avg_balance, transfers=None, credit_top3=None):
    full_date = extract_month_full_str(df["date"]) if df is not None and "date" in df.columns else ""
    # ... existing handlers ...

    if product == "Мультивалютный депозит":
        msg = (
            f"{name}, вы храните средства в разных валютах. "
            f"Мультивалютный депозит позволит выгодно накопить и получить дополнительный доход ≈{fmt_amount(benefit)} в месяц. "
            "Открыть мультивалютный депозит."
        )
        return truncate_msg(msg)

    # ... existing default message ...
    msg = f"{name}, рекомендуем {product}. Посмотреть в приложении."
    return truncate_msg(msg)

def truncate_msg(msg, max_len=220):
    # Ensure message within max length, keep whole last word
    if len(msg) <= max_len:
        return msg
    else:
        truncated = msg[:max_len]
        # Trim to last space to avoid cutting words
        truncated = truncated.rsplit(" ", 1)[0]
        # Add period if missing
        if not truncated.endswith(".") and not truncated.endswith("!"):
            truncated += "."
        return truncated


# =====================
# MAIN
# =====================
def run():
    base = CONFIG["BASE_DIR"]
    out_path = os.path.join(base, CONFIG["OUTPUT_FILE"])
    fnames = os.listdir(base)
    tx_pat = re.compile(r"client_(\d+)_transactions_3m\.csv", re.I)
    tr_pat = re.compile(r"client_(\d+)_transfers_3m\.csv", re.I)

    ids = {int(m.group(1)) for f in fnames if (m := tx_pat.match(f))}
    ids |= {int(m.group(1)) for f in fnames if (m := tr_pat.match(f))}
    print("Found client IDs:", len(ids))

    clients = safe_read_csv(os.path.join(base, "clients.csv"))
    if clients is not None and "client_code" in clients.columns:
        clients["client_code"] = pd.to_numeric(clients["client_code"], errors="coerce")

    results = []
    for cid in sorted(ids):
        tx = safe_read_csv(os.path.join(base, f"client_{cid}_transactions_3m.csv"))
        if tx is None:
            tx = pd.DataFrame()
        tr = safe_read_csv(os.path.join(base, f"client_{cid}_transfers_3m.csv"))
        if tr is None:
            tr = pd.DataFrame()

        tx = to_num(tx, "amount")
        tr = to_num(tr, "amount")

        row = None
        if clients is not None and "client_code" in clients.columns:
            sel = clients[clients["client_code"] == cid]
            if not sel.empty:
                row = sel.iloc[0].to_dict()

        name = row.get("name") if row else f"Клиент {cid}"
        avg_bal = float(row.get("avg_monthly_balance_KZT") or 0) if row else 0
        status = row.get("status") if row else ""

        scores, credit_top3 = score_all(tx, tr, avg_bal, status)

        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        # Pick the single most beneficial product, even if zero
        top_product, top_val = sorted_scores[0]

        push = generate_push(name, top_product, tx, top_val, avg_bal, transfers=tr, credit_top3=credit_top3)

        results.append({
            "client_code": cid,
            "product": top_product,
            "push_notification": push
        })

        print(f"client {cid}: {top_product} ({top_val:.1f})")

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved to", out_path)


if __name__ == "__main__":
    run()
