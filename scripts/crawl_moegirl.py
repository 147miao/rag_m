import json
import re
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

OUT = Path('/run/media/myj/D/rag_m/data/naruto')
OUT.mkdir(parents=True, exist_ok=True)
TITLES = ['火影忍者','漩涡鸣人','宇智波佐助','春野樱','旗木卡卡西','宇智波鼬','宇智波斑','千手柱间','大蛇丸','自来也','纲手','日向雏田','我爱罗','晓','木叶隐村','查克拉','尾兽','九尾','写轮眼','轮回眼','火影','火影忍者疾风传','宇智波带土','佩恩','长门','小南','赤砂之蝎','迪达拉','干柿鬼鲛','飞段','角都','药师兜','波风水门','漩涡玖辛奈','千手扉间','猿飞日斩','志村团藏','佐井','洛克李','天天','犬冢牙','油女志乃','秋道丁次','奈良鹿丸','山中井野','日向宁次','大筒木辉夜','六道仙人','十尾','须佐能乎','影分身之术','螺旋丸','千鸟','通灵之术','忍者联军','第四次忍界大战','妙木山','龙地洞','湿骨林']
JUNK = re.compile(r'^(登录|创建账号|首页|分类|讨论|查看源代码|查看历史|编辑|刷新|随机条目|帮助|关于|免责声明|隐私政策|导航|搜索|萌娘百科|工具|其他|目录|展开|收起)')

def clean(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 4 or JUNK.match(line):
            continue
        out.append(line)
    return '\n'.join(out)

def main():
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0')
        for t in TITLES:
            url = 'https://mzh.moegirl.org.cn/' + urllib.parse.quote(t)
            try:
                pg.goto(url, wait_until='domcontentloaded', timeout=30000)
                pg.wait_for_timeout(1800)
                text = clean(pg.inner_text('body'))
                if len(text) < 600:
                    print('[skip]', t, len(text), flush=True)
                    continue
                rows.append({'title': t, 'url': url, 'text': text, 'chars': len(text)})
                print('[ok]', t, len(text), flush=True)
            except PlaywrightError as e:
                print('[err]', t, type(e).__name__, str(e)[:60], flush=True)
            time.sleep(0.3)
        b.close()
    (OUT / 'moegirl.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print('done', len(rows), 'docs', sum(r['chars'] for r in rows)//1000, 'k chars')

main()
