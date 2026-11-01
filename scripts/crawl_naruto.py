"""从 360 百科抓取火影忍者相关资料（真实网页文档）。"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

OUT = Path("/run/media/myj/D/rag_m/data/naruto")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0"}
SEEDS = [
    "火影忍者", "漩涡鸣人", "宇智波佐助", "春野樱", "旗木卡卡西", "宇智波鼬", "宇智波斑", "千手柱间",
    "大蛇丸", "自来也", "纲手", "日向雏田", "我爱罗", "晓", "木叶村", "查克拉", "尾兽", "九尾",
    "写轮眼", "轮回眼", "火影", "火影忍者疾风传", "宇智波带土", "佩恩", "长门", "小南", "赤砂之蝎",
    "迪达拉", "干柿鬼鲛", "飞段", "角都", "药师兜", "波风水门", "漩涡玖辛奈", "千手扉间", "猿飞日斩",
    "志村团藏", "佐井", "洛克李", "天天", "犬冢牙", "油女志乃", "秋道丁次", "奈良鹿丸", "山中井野",
    "日向宁次", "大筒木辉夜", "六道仙人", "十尾", "须佐能乎", "影分身之术", "螺旋丸", "千鸟",
    "通灵之术", "忍者联军", "第四次忍界大战", "妙木山", "龙地洞", "湿骨林",
]
BOILER = re.compile(r"免费编辑|添加义项|义项指多义词|查看详细规范|所属类别|编辑分类|词条图册|参考资料|目录\s*\d")


def clean(text: str, title: str) -> str:
    text = re.sub(r"\s+", " ", text)
    idx = text.find(title)
    if idx > 0:
        text = text[idx:]
    text = BOILER.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def doc_links(soup: BeautifulSoup) -> list[str]:
    out: list[str] = []
    for a in soup.select("a"):
        href = a.get("href") or ""
        if "/doc/" not in href or "search" in href:
            continue
        if href.startswith("/"):
            href = "https://baike.so.com" + href
        if href not in out:
            out.append(href)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[str] = set()
    with httpx.Client(timeout=20, follow_redirects=True, headers=UA) as client:
        for seed in SEEDS:
            try:
                resp = client.get("https://baike.so.com/search", params={"q": seed})
                soup = BeautifulSoup(resp.text, "html.parser")
                links = doc_links(soup)[:2]
            except Exception as exc:  # noqa: BLE001
                print(f"[{seed}] search failed {type(exc).__name__}", flush=True)
                continue
            for url in links:
                if url in seen:
                    continue
                seen.add(url)
                try:
                    page = client.get(url)
                    if page.status_code != 200:
                        continue
                    soup = BeautifulSoup(page.text, "html.parser")
                    title = (soup.title.get_text(strip=True) if soup.title else "").replace("_360百科", "").strip()
                    best = None
                    for div in soup.find_all("div"):
                        t = div.get_text(" ", strip=True)
                        if len(t) > 1500 and (best is None or len(t) > len(best)):
                            best = t
                    if not best or len(best) < 800:
                        continue
                    text = clean(best, title)
                    if len(text) < 500:
                        continue
                    rows.append({"url": url, "title": title, "text": text, "chars": len(text)})
                    print(f"[{seed}] {title[:24]} {len(text)} chars", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{seed}] page failed {type(exc).__name__}", flush=True)
                time.sleep(0.4)
    (OUT / "raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"done: {len(rows)} docs, {sum(r['chars'] for r in rows)/1000:.1f}k chars", flush=True)


if __name__ == "__main__":
    main()
