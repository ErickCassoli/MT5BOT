import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from core.logging import get_logger

log = get_logger(__name__)

def find_latest_csv(logs_dir: Path) -> Path | None:
    files = sorted(logs_dir.glob("session_*.csv"))
    return files[-1] if files else None

def load_summary_json(csv_path: Path) -> dict | None:
    json_path = csv_path.with_suffix(".json")
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    # fallback: tentar parsear a linha de summary do CSV (menos rico)
    try:
        df = pd.read_csv(csv_path)
        row = df[df["event"]=="summary"].tail(1)
        if row.empty: return None
        txt = str(row["extra"].values[0])
        return {"text_summary": txt}
    except Exception:
        return None

def plot_orders_by_symbol(df: pd.DataFrame, outdir: Path):
    dfo = df[(df["event"]=="order") & (df["retcode"].notna())]
    if dfo.empty: return None
    series = dfo["symbol"].value_counts().sort_index()
    ax = series.plot(kind="bar", title="Orders by Symbol")
    ax.set_xlabel("Symbol"); ax.set_ylabel("Count")
    fig = ax.get_figure()
    out = outdir / "orders_by_symbol.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out

def plot_signals_per_hour(df: pd.DataFrame, outdir: Path):
    dfs = df[df["event"]=="signal"].copy()
    if dfs.empty: return None
    dfs["ts"] = pd.to_datetime(dfs["ts"], errors="coerce")
    dfs["hour"] = dfs["ts"].dt.hour
    piv = dfs.pivot_table(index="hour", columns="symbol", values="event", aggfunc="count").fillna(0)
    ax = piv.plot(kind="line", marker="o", title="Signals per Hour (UTC)")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Signals")
    fig = ax.get_figure()
    out = outdir / "signals_per_hour.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out

def plot_nearbreak_hist(df: pd.DataFrame, outdir: Path):
    dfs = df[df["event"]=="signal"].copy()
    if dfs.empty: return None
    # distância mínima até canal em fração do near_thr
    for col in ["dist_up","dist_low","near_thr"]:
        dfs[col] = pd.to_numeric(dfs[col], errors="coerce")
    frac = []
    for _, r in dfs.iterrows():
        if pd.notna(r["near_thr"]) and r["near_thr"]>0:
            d = min(x for x in [r["dist_up"], r["dist_low"]] if pd.notna(x))
            frac.append(d / r["near_thr"])
    if not frac: return None
    fig, ax = plt.subplots()
    ax.hist(frac, bins=20)
    ax.set_title("Near-break distance (fraction of threshold)")
    ax.set_xlabel("distance / near_thr"); ax.set_ylabel("count")
    out = outdir / "nearbreak_hist.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out

def build_markdown(summary: dict | None, imgs: list[Path], csv_path: Path) -> str:
    lines = ["# Session Report", ""]
    lines.append(f"- **CSV**: `{csv_path.name}`")
    if summary:
        if "text_summary" in summary:
            lines.append(f"- **Summary**: {summary['text_summary']}")
        else:
            lines.append(f"- **Started**: {summary.get('started_at')}")
            lines.append(f"- **Ended**: {summary.get('ended_at')}")
            lines.append(f"- **Baseline**: {summary.get('baseline')}")
            lines.append(f"- **Equity Now**: {summary.get('equity_now')}")
            lines.append(f"- **Gain %**: {summary.get('gain_pct')}")
            lines.append(f"- **Realized PnL**: {summary.get('realized_pnl')}")
            lines.append(f"- **Closed Trades**: {summary.get('closed_trades')}  (Wins: {summary.get('wins')} | Losses: {summary.get('losses')})")
            pf = summary.get('profit_factor')
            if pf is not None: lines.append(f"- **Profit Factor**: {pf}")
            if summary.get("symbols"):
                lines.append(f"- **Symbols**: {', '.join(summary['symbols'])}")
    lines.append("")
    for img in imgs:
        if img: lines.append(f"![{img.stem}]({img.name})")
    return "\n".join(lines)

def save_markdown_and_html(md: str, outdir: Path):
    md_path = outdir / "report.md"
    md_path.write_text(md, encoding="utf-8")
    # html simples
    html = f"<!doctype html><meta charset='utf-8'><style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px;max-width:900px}}img{{max-width:100%}}</style>\n{md.replace('\n','<br>')}"
    html_path = outdir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return md_path, html_path

def main():
    ap = argparse.ArgumentParser(description="Analyze FXBot session CSV/JSON and generate charts/report.")
    ap.add_argument("--logs-dir", default=str(Path(__file__).resolve().parents[1] / "logs"))
    ap.add_argument("--csv", default=None, help="Path to a specific session_*.csv")
    ap.add_argument("--latest", action="store_true", help="Use the latest CSV in logs dir")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    csv_path = Path(args.csv) if args.csv else (find_latest_csv(logs_dir) if args.latest else None)
    if not csv_path or not csv_path.exists():
        raise SystemExit("CSV não encontrado. Use --csv <arquivo> ou --latest.")

    # output dir p/ relatórios
    ts = csv_path.stem.replace("session_", "")
    outdir = logs_dir / f"report_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    summary = load_summary_json(csv_path)

    imgs = []
    imgs.append(plot_orders_by_symbol(df, outdir))
    imgs.append(plot_signals_per_hour(df, outdir))
    imgs.append(plot_nearbreak_hist(df, outdir))

    md = build_markdown(summary, [i for i in imgs if i], csv_path)
    md_path, html_path = save_markdown_and_html(md, outdir)
    log.info(f"Report gerado em:\n - {md_path}\n - {html_path}")

if __name__ == "__main__":
    main()
