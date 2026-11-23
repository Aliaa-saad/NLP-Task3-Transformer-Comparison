"""End-to-end UI check + screenshots with Playwright.

Usage (backend venv, server already running):
    python app/tools/e2e_screenshots.py --url http://localhost:8000 --out app/screenshots [--prefix real] [--with-comparison]

Uses the locally installed Chrome (no browser download). Exits non-zero on console errors
or failed checks.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# Team rule: no real Yelp review text in committed screenshots. The error-analysis card (the only
# place real dataset reviews appear) is hidden before every screenshot, and each screenshot is
# checked against the error-analysis CSV so no real review snippet is visible.
HIDE_REVIEW_TEXT_JS = """
() => {
  for (const h of document.querySelectorAll('h2')) {
    if (h.textContent.includes('highest-confidence errors')) {
      const card = h.closest('section');
      if (card) card.style.display = 'none';
    }
  }
}
"""


# Public-sharing rule: no course/assignment wording in any screenshot.
ACADEMIC_WORDS = re.compile(
    r"\b(tasks?|courses?|assignments?|homework|rubrics?|students?|lectures?|universit\w*|mandatory|academic|"
    r"non-commercial|licen[cs]e|agreement|graded)\b",
    re.I,
)


def load_review_snippets() -> list[str]:
    path = Path(__file__).resolve().parents[1] / "backend" / "artifacts" / "model_a" / "error_analysis_top50.csv"
    if not path.exists():
        return []
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        texts = [row.get("text_or_index", "") for row in csv.DictReader(f)]
    # first ~30 chars of each first line; long enough to be unique, short enough to survive clamping
    return [t.strip().splitlines()[0][:30] for t in texts if t and len(t.strip()) >= 12]


# A missing optional artifact legitimately yields an HTTP 404, which Chrome logs as a resource error.
EXPECTED_404 = ("/api/metrics",)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="app/screenshots")
    ap.add_argument("--prefix", default="real")
    ap.add_argument("--with-comparison", dest="sample", action="store_true", help="server has comparison_results.json: exercise the full Compare dashboard")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = next((p for p in CHROME_CANDIDATES if Path(p).exists()), None)

    problems: list[str] = []
    console_errors: list[str] = []
    expected_404s: list[str] = []

    snippets = load_review_snippets()
    print(f"review-text guard: {len(snippets)} snippets from error_analysis_top50.csv")

    def shot(page: Page, name: str, full: bool = True) -> None:
        page.evaluate(HIDE_REVIEW_TEXT_JS)
        visible = page.evaluate("document.body.innerText")
        leaked = [sn for sn in snippets if sn in visible]
        words = sorted({m.group(0).lower() for m in ACADEMIC_WORDS.finditer(visible)})
        if words:
            problems.append(f"{name}: academic wording visible: {words}")
            print("  FAIL academic wording visible:", words)
        path = out / f"{args.prefix}_{name}.png"
        if leaked:
            problems.append(f"{path.name}: real review text visible ({leaked[0]!r}...); not saved")
            print("  FAIL review text visible, screenshot NOT saved:", path.name)
            return
        page.screenshot(path=str(path), full_page=full)
        print("  saved", path, "(no review text)")

    def check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            problems.append(msg)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=exe, headless=True)
        passes = [
            ("desktop", {"width": 1440, "height": 900}, "dark"),
            ("mobile", {"width": 390, "height": 844}, "dark"),
            ("desktop", {"width": 1440, "height": 900}, "light"),
            ("mobile", {"width": 390, "height": 844}, "light"),
        ]
        for vp_name, vp, theme in passes:
            tag = vp_name if theme == "dark" else f"{vp_name}_light"
            dark = theme == "dark"
            print(f"\n== {vp_name} {vp} · {theme}")
            ctx = browser.new_context(viewport=vp, device_scale_factor=1 if vp_name == "desktop" else 2, reduced_motion="no-preference")
            ctx.add_init_script(f"try {{ localStorage.setItem('theme', '{theme}'); }} catch (e) {{}}")
            page = ctx.new_page()
            page.on("console", lambda m, t=tag: console_errors.append(f"[{t}] {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e, t=tag: console_errors.append(f"[{t}] pageerror: {e}"))
            page.on(
                "response",
                lambda r: expected_404s.append(r.url) if r.status == 404 and any(p in r.url for p in EXPECTED_404) else None,
            )

            # ---------------- Analyze
            page.goto(args.url, wait_until="networkidle")
            check(page.evaluate("document.documentElement.dataset.theme") == theme, f"{theme} theme applied")
            check(page.get_by_role("heading", name="Two Transformers. One review.").is_visible(), "hero headline visible")
            ticker = page.get_by_role("region", name=re.compile("Live inference ticker"))
            check(ticker.is_visible(), "live review ticker visible")
            expect(ticker.get_by_text("A →")).to_be_visible(timeout=20_000)
            check(ticker.get_by_text("B →").is_visible(), "ticker shows A/B chips from real predictions")
            check(page.get_by_text("3-class sentiment analysis of Yelp reviews").is_visible(), "overview visible")
            check(page.get_by_text("Sentiment · Transformers").first.is_visible(), "hero eyebrow visible")
            chips = page.get_by_role("group", name="Try an example").get_by_role("button")
            check(chips.count() >= 5, f"{chips.count()} example chips")
            if dark:
                shot(page, f"{tag}_analyze_empty")

                # client-side validation
                page.get_by_role("button", name="Run both models").click()
                check(page.get_by_role("alert").filter(has_text="empty").is_visible(), "empty-text validation shown inline")
                page.get_by_role("textbox", name="Your review").fill("x" * 5001)
                page.get_by_role("button", name="Run both models").click()
                check(page.get_by_text("too long").first.is_visible(), "5,000-char limit validation shown")

            # prediction with an expected label
            expected_group = page.get_by_role("radiogroup", name="Expected label (optional)")
            check(expected_group.get_by_role("radio").count() == 4, "expected-label control: none/negative/neutral/positive")
            expected_group.get_by_role("radio", name="Positive").click()
            chips.filter(has_text="Clearly positive").click()
            page.get_by_role("button", name="Run both models").click()
            expect(page.get_by_role("status").filter(has_text="agree").first).to_be_visible(timeout=30_000)
            page.wait_for_timeout(900)
            cards = page.locator("article[aria-label$='result']")
            check(cards.count() == 2, "two prediction cards")
            body = page.locator("main").inner_text()
            check("latency" in body and "ms" in body, "latency shown on cards")
            check("round-trip" in body, "round-trip time shown separately")
            check("Both models agree" in body and "Positive" in body, "clearly positive -> both agree positive")
            check(cards.locator("[data-testid=verdict-badge][data-correct=true]").count() == 2, "both cards marked ✓ correct")
            check("Both models got it right" in page.get_by_test_id("verdict").inner_text(), "verdict line shown")
            check("Compare page" in page.get_by_test_id("verdict").inner_text(), "verdict points to Compare for overall performance")
            check(ticker.get_by_text("real prediction for your input").is_visible(), "ticker shows the real result for the user's input")
            shot(page, f"{tag}_analyze_result")

            if vp_name == "desktop" and dark:
                chips.filter(has_text="Long").click()
                page.get_by_role("button", name="Run both models").click()
                expect(page.get_by_role("status").filter(has_text="agree").first).to_be_visible(timeout=30_000)
                page.wait_for_timeout(900)
                check("truncated (" in page.locator("main").inner_text(), "long review flagged as truncated")
                verdict = page.get_by_test_id("verdict").inner_text()
                check("got it right" in verdict or "missed" in verdict, f"verdict for long review: {verdict.splitlines()[0]}")
                shot(page, f"{tag}_analyze_long_truncated")

                expected_group.get_by_role("radio", name="None").click()
                check(page.get_by_test_id("verdict").count() == 0, "expected = none -> no verdict/badges")
                check(cards.locator("[data-testid=verdict-badge]").count() == 0, "no ✓/✗ badges with none")

                switch = page.get_by_role("switch", name="Apply tuned logit biases (Model A)")
                switch.focus()
                page.keyboard.press("Space")
                check(not switch.is_checked(), "bias toggle switches off via keyboard")
                page.get_by_text("Apply tuned logit biases (Model A)").click()
                check(switch.is_checked(), "bias toggle switches back on via mouse")

            # ---------------- Compare
            page.get_by_role("link", name="Compare", exact=True).click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1200)
            if args.sample:
                expect(page.get_by_text("Δ Macro-F1 (B − A)")).to_be_visible(timeout=15_000)
                check(page.get_by_role("table").count() >= 4, "metrics + efficiency tables + 2 confusion matrices")
                check(page.get_by_text("Bootstrap").first.is_visible(), "bootstrap sentence visible")
                eff = page.get_by_role("heading", name="Size, speed and training cost")
                check(eff.is_visible(), "efficiency section visible")
                main_text = page.locator("main").inner_text()
                check("scratch training (A100)" in main_text and "fine-tuning (T4)" in main_text, "training regime + GPU labels")
                check("Hardware differs" in main_text, "hardware-differs note")
                check(page.get_by_role("heading", name="Why the difference?").is_visible(), "why-the-difference section visible")
                check("statistically tied" in main_text and "Neutral recall" in main_text, "generated explanation (natural tie + neutral)")
                check("Negative precision" in main_text and "Neutral recall" in main_text, "per-class precision/recall in metrics table")
                fits = page.evaluate(
                    "() => [...document.querySelectorAll('table caption.sr-only')].map(c => c.closest('div'))"
                    ".every(d => !d || d.scrollWidth <= d.clientWidth + 1)"
                )
                check(fits, "tables fit without horizontal scrolling")
                shot(page, f"{tag}_compare_balanced")
                if dark:
                    page.get_by_role("radio", name="Natural").click()
                    page.wait_for_timeout(1000)
                    check(page.get_by_role("radio", name="Natural").get_attribute("aria-checked") == "true", "Natural view selected")
                    shot(page, f"{tag}_compare_natural")
                    page.get_by_role("radio", name="A: raw").click()
                    page.wait_for_timeout(500)
                    check(page.get_by_text("Model A (raw)").first.is_visible(), "raw/adjusted toggle works")
                    page.get_by_role("radio", name="Balanced").click()
                    page.get_by_role("radio", name="A: logit-adjusted").click()
            else:
                check(page.get_by_text("Comparison results not available").is_visible(), "missing comparison file -> clear message")
                shot(page, f"{tag}_compare")

            # ---------------- About
            page.get_by_role("link", name="About", exact=True).click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(900)
            check(page.get_by_role("heading", name="Star rating → sentiment").is_visible(), "about page label mapping")
            check(page.get_by_role("heading", name="Dataset & setup").is_visible(), "dataset & setup card")
            about = page.locator("main").inner_text()
            check("kaggle.com/datasets/yelp-dataset/yelp-dataset" in about and "License" not in about, "dataset URL shown, no license line")
            check("vocabulary only" in about and "not pretrained weights" in about, "Model A tokenizer-only note")
            shot(page, f"{tag}_about")

            sw = page.evaluate("document.documentElement.scrollWidth")
            check(sw <= vp["width"], f"no horizontal scroll (scrollWidth={sw})")
            ctx.close()

        # ---------------- reduced motion: ticker is static (no typing cursor, no scan line)
        print("\n== reduced motion")
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        page = ctx.new_page()
        page.on("console", lambda m: console_errors.append(f"[reduced] {m.text}") if m.type == "error" else None)
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        check(page.locator(".ticker-cursor").count() == 0 and page.locator(".ticker-scan").count() == 0, "reduced motion: no typing cursor / scan line")
        check(len(page.get_by_role("region", name=re.compile("Live inference ticker")).inner_text()) > 40, "reduced motion: static review text shown")
        ctx.close()
        browser.close()

    real_console = [
        e for e in console_errors
        if not ("Failed to load resource" in e and "404" in e and expected_404s)
    ]
    print("\nconsole errors:", real_console or "none")
    if expected_404s:
        print(f"(ignored {len(console_errors) - len(real_console)} expected 404 resource logs for missing optional artifacts)")
    if problems or real_console:
        print("\nFAILED:", problems + real_console)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
