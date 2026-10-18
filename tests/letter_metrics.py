"""Letter metrics: sign every letter, several tries each, and find the weakest ones.

    .venv/bin/python tests/letter_metrics.py                          # 24 letters x 3 tries x 5 s
    .venv/bin/python tests/letter_metrics.py --letters RUKHN --tries 2 --seconds 6
    .venv/bin/python tests/letter_metrics.py --source uart            # parse the live UART line instead
    .venv/bin/python tests/letter_metrics.py --replay tests/out/metrics-<stamp>.json   # redo the tables

For each letter the script says "Sign A: hold it in view for 15 s and press Enter"; you press
Enter once and hold the sign while it samples the board for --tries x --seconds in one go,
printing every sample as it arrives:

    A 0.91 p0.97 box 112,84 96px R 15.8 fps

The window is split into --tries consecutive segments ("tries"), so a letter that reads
right at first and drifts later shows up as FLAKY. Per try it scores: top-1 majority (hit),
share of frames reading the letter, median confidence
on those frames, median hand-box width in px (the distance proxy: < 75 px dropped out in the
signing sessions), "no hand" share, frames read as a left hand, and letters committed to the
text during the window. Per letter it aggregates the tries; the final table is sorted weakest
first, followed by a row-normalised confusion matrix (signed vs read, per frame, with a
"no hand" column) and scikit-learn's classification_report over the frames (a built-in
fallback prints the same layout without scikit-learn). All of it is written to
tests/out/metrics-<stamp>.json and .md.

Source: the board's telemetry over J-Link (`g_fs_tele`, no UART needed, ~2-4 samples/s; each
read halts the core for a few ms) or the live UART line (`--source uart`, every frame; needs
SW4 on UART4 and the VS Code Serial Monitor closed). `--source auto` (default) tries the UART
for two seconds and falls back to J-Link.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_tools as fs  # noqa: E402

LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY"  # the 24 static letters; J and Z are motion letters
OUT = fs.REPO / "tests" / "out"
MIN_BOX_PX = 75  # below this the landmark presence gate dropped out in the signing sessions
# Live UART lines (src/app_main.cpp): with and without a hand.
LIVE = re.compile(r"^([A-Y]) (\d\.\d\d) [A-Y] \d\.\d\d [A-Y] \d\.\d\d \| box (-?\d+),(-?\d+) (\d+)x\d+ s[\d.]+ p([\d.]+) "
                  r"\| .* \| ([\d.]+) fps$", re.M)
NO_HAND = re.compile(r"^no hand \| .* \| ([\d.]+) fps$", re.M)


# ---- samples ---------------------------------------------------------------------------

def sample_from_tele(t: list[int]) -> dict:
    """One telemetry read (tests/board_telemetry.py layout) -> a sample dict."""
    return {"t": time.time(), "letter": chr(t[0]) if t[0] else None, "prob": t[1] / 1000, "presence": t[2] / 1000,
            "fps": t[3] / 1000, "box": [t[6], t[7], t[8]], "pending": chr(t[9]) if t[9] else None,
            "text_len": t[11], "right": t[12] >= 500, "hand_frames": t[13]}


class JLinkSource:
    name = "jlink"

    def __init__(self) -> None:
        from board_telemetry import read_tele  # noqa: F401  (tests/board_telemetry.py)
        self.read_tele = read_tele
        self.read_tele()  # fail early if the board is unreachable

    def window(self, seconds: float, on_sample) -> list[dict]:
        samples, end = [], time.time() + seconds
        while time.time() < end:
            s = sample_from_tele(self.read_tele())
            samples.append(s)
            on_sample(s)
        return samples


class UartSource:
    name = "uart"

    def __init__(self) -> None:
        self.uart = fs.UartCapture()
        self.uart.read_for(2.0)
        if not (LIVE.search(self.uart.text) or NO_HAND.search(self.uart.text)):
            self.uart.close()
            raise RuntimeError("no live lines on the UART in 2 s")

    def window(self, seconds: float, on_sample) -> list[dict]:
        self.uart.data.clear()
        samples, end, seen = [], time.time() + seconds, 0
        while time.time() < end:
            self.uart.read_for(0.25)
            lines = self.uart.text.splitlines()
            for line in lines[seen:]:
                s = parse_live(line)
                if s:
                    samples.append(s)
                    on_sample(s)
            seen = len(lines)
        return samples

    def close(self) -> None:
        self.uart.close()


def parse_live(line: str) -> dict | None:
    m = LIVE.match(line.strip())
    if m:
        return {"t": time.time(), "letter": m.group(1), "prob": float(m.group(2)), "presence": float(m.group(6)),
                "fps": float(m.group(7)), "box": [int(m.group(3)), int(m.group(4)), int(m.group(5))],
                "pending": None, "text_len": None, "right": None, "hand_frames": None}
    m = NO_HAND.match(line.strip())
    if m:
        return {"t": time.time(), "letter": None, "prob": 0.0, "presence": 0.0, "fps": float(m.group(1)),
                "box": [0, 0, 0], "pending": None, "text_len": None, "right": None, "hand_frames": None}
    return None


def open_source(kind: str):
    if kind in ("auto", "uart"):
        try:
            src = UartSource()
            print("source: UART live line")
            return src
        except (fs.UartBusy, RuntimeError, OSError) as e:
            if kind == "uart":
                raise
            print(f"UART not usable ({e}); using J-Link telemetry")
    src = JLinkSource()
    print("source: g_fs_tele over J-Link")
    return src


# ---- scoring ---------------------------------------------------------------------------

def score_try(letter: str, samples: list[dict]) -> dict:
    n = len(samples)
    hand = [s for s in samples if s["letter"]]
    votes = Counter(s["letter"] for s in hand)
    top, top_n = votes.most_common(1)[0] if votes else (None, 0)
    right = [s["prob"] for s in hand if s["letter"] == letter]
    boxes = [s["box"][2] for s in hand if s["box"][2] > 0]
    lefts = [s for s in hand if s["right"] is False]
    commits = None
    if samples and samples[0]["text_len"] is not None:
        commits = samples[-1]["text_len"] - samples[0]["text_len"]
        if commits < 0:  # the decoder's text wrapped or was cleared
            commits = None
    return {
        "letter": letter, "samples": n, "hand_samples": len(hand), "read_as": top, "hit": top == letter,
        "share": len(right) / n if n else 0.0, "top_share": top_n / n if n else 0.0,
        "median_conf": statistics.median(right) if right else 0.0,
        "median_box_px": statistics.median(boxes) if boxes else 0,
        "no_hand_share": 1 - len(hand) / n if n else 1.0,
        "left_share": len(lefts) / len(hand) if hand and samples[0]["right"] is not None else None,
        "commits": commits, "fps": statistics.median(s["fps"] for s in samples) if samples else 0.0,
        "votes": dict(votes.most_common(5)),
    }


def score_letter(letter: str, tries: list[dict]) -> dict:
    confs = [t["median_conf"] for t in tries if t["median_conf"] > 0]
    confusions: Counter = Counter()
    for t in tries:
        for k, v in t["votes"].items():
            if k != letter:
                confusions[k] += v
    return {
        "letter": letter, "tries": len(tries), "hits": sum(t["hit"] for t in tries),
        "share": statistics.mean(t["share"] for t in tries) if tries else 0.0,
        "median_conf": statistics.median(confs) if confs else 0.0,
        "median_box_px": statistics.median(t["median_box_px"] for t in tries) if tries else 0,
        "no_hand_share": statistics.mean(t["no_hand_share"] for t in tries) if tries else 1.0,
        "left_share": (statistics.mean(t["left_share"] for t in tries if t["left_share"] is not None)
                       if any(t["left_share"] is not None for t in tries) else None),
        "commits": [t["commits"] for t in tries],
        "confused_with": dict(confusions.most_common(3)),
    }


def weakness(l: dict) -> tuple:
    """Sort key: fewest hits, lowest share of correct frames, lowest confidence."""
    return (l["hits"] / l["tries"] if l["tries"] else 0.0, l["share"], l["median_conf"])


def verdict(l: dict) -> str:
    if l["tries"] and l["hits"] == 0:
        return "FAIL " + (f"reads as {', '.join(l['confused_with'])}" if l["confused_with"] else "no hand")
    if l["hits"] < l["tries"]:
        return "FLAKY"
    if l["median_conf"] < 0.7:
        return "LOW CONF"
    return "ok"


NO_HAND_LABEL = "none"


def predictions(raw: dict[str, list[list[dict]]]) -> tuple[list[str], list[str]]:
    """Every sample as (signed letter, letter read); no hand reads as NO_HAND_LABEL."""
    y_true, y_pred = [], []
    for letter, tries in raw.items():
        for samples in tries:
            for s in samples:
                y_true.append(letter)
                y_pred.append(s["letter"] or NO_HAND_LABEL)
    return y_true, y_pred


def confusion_table(y_true: list[str], y_pred: list[str]) -> str:
    """Row-normalised confusion matrix (rows: signed, columns: read), in percent."""
    rows = sorted(set(y_true))
    cols = rows + sorted(set(y_pred) - set(rows) - {NO_HAND_LABEL}) + ([NO_HAND_LABEL] if NO_HAND_LABEL in y_pred else [])
    try:
        from sklearn.metrics import confusion_matrix

        m = confusion_matrix(y_true, y_pred, labels=cols, normalize="true")
        matrix = {r: {c: float(m[i][j]) for j, c in enumerate(cols)} for i, r in enumerate(cols) if r in rows}
    except ImportError:
        counts = Counter(zip(y_true, y_pred))
        matrix = {}
        for r in rows:
            n = sum(counts[(r, c)] for c in cols)
            matrix[r] = {c: counts[(r, c)] / n if n else 0.0 for c in cols}
    head = "| signed \\ read | " + " | ".join(c if c != NO_HAND_LABEL else "no hand" for c in cols) + " |"
    lines = ["## Confusion matrix (row-normalised, % of frames)", "", head, "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        cells = []
        for c in cols:
            v = matrix[r][c]
            cells.append(f"**{100 * v:.0f}**" if c == r else (f"{100 * v:.0f}" if v >= 0.005 else "."))
        lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def classification_table(y_true: list[str], y_pred: list[str]) -> str:
    """scikit-learn's classification_report over the samples (built-in fallback, same layout)."""
    labels = sorted(set(y_true))
    try:
        from sklearn.metrics import classification_report

        text = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=2)
    except ImportError:
        text = _classification_report(y_true, y_pred, labels)
    return "## Classification report (per frame; precision counts frames read as the letter while another was signed)\n\n```\n" + text.rstrip() + "\n```"


def _classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> str:
    rows, n = [], len(y_true)
    tp = Counter(t for t, p in zip(y_true, y_pred) if t == p)
    pred = Counter(y_pred)
    true = Counter(y_true)
    for l in labels:
        p = tp[l] / pred[l] if pred[l] else 0.0
        r = tp[l] / true[l] if true[l] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((l, p, r, f, true[l]))
    acc = sum(tp.values()) / n if n else 0.0
    macro = [statistics.mean(r[i] for r in rows) for i in (1, 2, 3)]
    weighted = [sum(r[i] * r[4] for r in rows) / n for i in (1, 2, 3)] if n else [0.0] * 3
    w = max(12, max(len(l) for l in labels))
    out = [f"{'':{w}} precision    recall  f1-score   support", ""]
    out += [f"{l:>{w}} {p:9.2f} {r:9.2f} {f:9.2f} {s:9d}" for l, p, r, f, s in rows]
    out += ["", f"{'accuracy':>{w}} {'':9} {'':9} {acc:9.2f} {n:9d}",
            f"{'macro avg':>{w}} {macro[0]:9.2f} {macro[1]:9.2f} {macro[2]:9.2f} {n:9d}",
            f"{'weighted avg':>{w}} {weighted[0]:9.2f} {weighted[1]:9.2f} {weighted[2]:9.2f} {n:9d}"]
    return "\n".join(out)


def tables(letters: list[dict], source: str, seconds: float, raw: dict | None = None) -> str:
    rows = sorted(letters, key=weakness)
    lines = [f"# Letter metrics ({source}, {seconds:.0f} s per try)", "",
             "| Letter | Hits | Correct frames | Median conf | Box px | No hand | Left | Commits/try | Confused with | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for l in rows:
        left = f"{100 * l['left_share']:.0f}%" if l["left_share"] is not None else "-"
        commits = "/".join("?" if c is None else str(c) for c in l["commits"])
        conf = ", ".join(f"{k} {v}" for k, v in l["confused_with"].items()) or "-"
        lines.append(f"| {l['letter']} | {l['hits']}/{l['tries']} | {100 * l['share']:.0f}% | {l['median_conf']:.2f} | "
                     f"{l['median_box_px']:.0f} | {100 * l['no_hand_share']:.0f}% | {left} | {commits} | {conf} | {verdict(l)} |")
    weakest = [l for l in rows if verdict(l) != "ok"][:5]
    lines += ["", "Weakest: " + (", ".join(f"{l['letter']} ({verdict(l)})" for l in weakest) if weakest else "none"),
              f"Letters with a hit on every try: {sum(l['hits'] == l['tries'] for l in rows)}/{len(rows)}; "
              f"median confidence over letters: {statistics.median(l['median_conf'] for l in rows):.2f}" if rows else ""]
    small = [l["letter"] for l in rows if 0 < l["median_box_px"] < MIN_BOX_PX]
    if small:
        lines.append(f"Hand box under {MIN_BOX_PX} px (too far from the camera, expect dropouts): {', '.join(small)}")
    if raw:
        y_true, y_pred = predictions(raw)
        if y_true:
            lines += ["", confusion_table(y_true, y_pred), "", classification_table(y_true, y_pred)]
    return "\n".join(lines)


# ---- main ------------------------------------------------------------------------------

def fmt(s: dict) -> str:
    if not s["letter"]:
        return f"no hand  p{s['presence']:.2f} {s['fps']:.1f} fps"
    hand = "" if s["right"] is None else (" R" if s["right"] else " L")
    pend = f" pending {s['pending']}" if s["pending"] else ""
    return (f"{s['letter']} {s['prob']:.2f} p{s['presence']:.2f} box {s['box'][0]},{s['box'][1]} {s['box'][2]}px"
            f"{hand} {s['fps']:.1f} fps{pend}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", default=LABELS, help=f"letters to test (default: the 24 static ones, {LABELS})")
    ap.add_argument("--tries", type=int, default=3, help="segments the letter's window is split into")
    ap.add_argument("--seconds", type=float, default=5.0, help="seconds per segment (the window is tries x seconds)")
    ap.add_argument("--source", choices=["auto", "jlink", "uart"], default="auto")
    ap.add_argument("--no-prompt", action="store_true", help="start each try after a 2 s countdown instead of Enter")
    ap.add_argument("--replay", type=Path, help="rebuild the tables from a saved metrics JSON, no board")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.replay:
        saved = json.loads(args.replay.read_text())
        print(tables(saved["letters"], saved["source"], saved["seconds"], saved.get("samples")))
        return 0

    letters = [c for c in args.letters.upper() if c in LABELS]
    src = open_source(args.source)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results: list[dict] = []
    raw: dict[str, list[list[dict]]] = {}
    total = args.seconds * args.tries
    try:
        for letter in letters:
            # One Enter per letter: one window of tries x seconds, split into `tries` segments
            # so the per-try numbers show whether the reading holds steady or drifts.
            if args.no_prompt:
                print(f"\nSign  {letter}  for {total:.0f} s ... starting in 2 s", flush=True)
                time.sleep(2.0)
            else:
                input(f"\nSign  {letter}: hold it in view for {total:.0f} s and press Enter... ")
            segment = [0]

            def on_sample(s, letter=letter):
                k = min(int((s["t"] - start) / args.seconds), args.tries - 1)
                if k != segment[0]:
                    segment[0] = k
                    print(f"  -- segment {k + 1}/{args.tries}", flush=True)
                print("  " + fmt(s), flush=True)

            start = time.time()
            samples = src.window(total, on_sample)
            if not samples:
                print("  no samples: is the board running the live loop?")
            segments = [[s for s in samples if min(int((s["t"] - start) / args.seconds), args.tries - 1) == k]
                        for k in range(args.tries)]
            tries = []
            for k, seg in enumerate(segments, 1):
                t = score_try(letter, seg)
                tries.append(t)
                raw.setdefault(letter, []).append(seg)
                print(f"  segment {k}: {'HIT ' if t['hit'] else 'MISS'} read as {t['read_as'] or '-'} in {100 * t['top_share']:.0f}% of "
                      f"{t['samples']} samples | {letter} {100 * t['share']:.0f}% conf {t['median_conf']:.2f} | box {t['median_box_px']:.0f} px "
                      f"| no hand {100 * t['no_hand_share']:.0f}%"
                      + (f" | commits {t['commits']}" if t["commits"] is not None else ""))
            l = score_letter(letter, tries)
            l["per_try"] = tries
            results.append(l)
            print(f"  {letter}: {l['hits']}/{l['tries']} tries, correct frames {100 * l['share']:.0f}%, "
                  f"median conf {l['median_conf']:.2f} -> {verdict(l)}")
    except KeyboardInterrupt:
        print("\ninterrupted: reporting what was measured")
    finally:
        if hasattr(src, "close"):
            src.close()

    if not results:
        return 2
    report = {"stamp": stamp, "source": src.name, "seconds": args.seconds, "tries": args.tries,
              "letters": results, "samples": raw}
    (OUT / f"metrics-{stamp}.json").write_text(json.dumps(report, indent=1))
    md = tables(results, src.name, args.seconds, raw)
    (OUT / f"metrics-{stamp}.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nwritten: tests/out/metrics-{stamp}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
