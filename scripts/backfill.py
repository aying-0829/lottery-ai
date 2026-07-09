"""
AI 彩票分析工具 — 全量历史回填脚本（一次性）
================================================
把双色球(2003-至今)、大乐透(2007-至今)的完整开奖历史从 500.com 的
datachart「newinc/history.php」接口一次拉取并写入 data/ 目录，覆盖/合并现有数据。

说明
----
- 500.com 的 `history.shtml` 对海外/普通请求只返回「最新30期」且忽略分页参数；
  但 `datachart.500.com/<game>/history/newinc/history.php?start=...&end=...`
  支持 start/end 范围参数，**可返回完整历史**（已实测：SSQ 3475 期 / DLT 2894 期）。
- 本脚本与 crawl.py 的 parse_500_rows / 字段结构保持一致，输出 schema 完全一致，
  下游 model.py / save_predict.py / backtest.py / 前端 无需任何改动。
- 写入策略：以回填数据为准，并保留现有文件中「回填未覆盖」的期号（防数据丢失）。
- 仅本地/一次性运行，不进入 CI（CI 仍用 crawl.py 做每日增量）。

用法
----
    python scripts/backfill.py
可选环境变量：
    BF_END  结束期号后缀（默认 '26999'，足够涵盖未来若干年）
"""

import json
import re
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ 配置 ============

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TZ = timezone(timedelta(hours=8))

SSQ_FILE = os.path.join(DATA_DIR, 'ssq_data.json')
DLT_FILE = os.path.join(DATA_DIR, 'dlt_data.json')

# 各彩种起始期号（最早一期）
SSQ_START = '03001'   # 2003-02-23 双色球第 03001 期
DLT_START = '07001'   # 2007-05-30 大乐透第 07001 期

# 合理性下限（低于此值说明抓取异常，应中止以免写入残缺数据）
SSQ_MIN_ROWS = 3300
DLT_MIN_ROWS = 2700

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')


# ============ HTTP ============

def create_session():
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
    )
    s.mount('https://', HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=5))
    return s


def fetch_full(session, game, start, end):
    """拉取某彩种全量历史（单请求，newinc 接口支持范围参数）。"""
    url = (f'https://datachart.500.com/{game}/history/'
           f'newinc/history.php?start={start}&end={end}')
    headers = {
        'User-Agent': UA,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
    }
    resp = session.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    resp.encoding = 'gb2312'
    return resp.text


# ============ 解析（与 crawl.py 保持一致）============

def parse_500_rows(html):
    """从 500.com 历史页 <tbody id="tdata"> 提取每行单元格文本列表。"""
    m = re.search(r'<tbody id="tdata">(.*?)</tbody>', html, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    rows = []
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.DOTALL):
        tr = re.sub(r'<!--.*?-->', '', tr, flags=re.DOTALL)
        cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)
        cleaned = [c.replace('&nbsp;', '').strip() for c in cells]
        if cleaned:
            rows.append(cleaned)
    return rows


def _first_date(cells):
    for c in cells:
        if re.match(r'\d{4}-\d{2}-\d{2}', c):
            return c
    return ''


def norm_ssq_period(p):
    """双色球期号统一为 7 位（5 位 → '20'+5 位）。"""
    p = p.strip()
    if re.match(r'^\d{5}$', p):
        return '20' + p
    return p


def build_ssq(rows):
    out = {}
    for cells in rows:
        if len(cells) < 9:
            continue
        period = norm_ssq_period(cells[0])
        if not re.match(r'^\d{7}$', period):
            continue
        red = [int(x) for x in cells[1:7] if x.isdigit()]
        blue = int(cells[7]) if cells[7].isdigit() else 0
        if len(red) != 6 or not (1 <= blue <= 16):
            continue
        out[period] = {
            'period': period,
            'date': _first_date(cells),
            'red': sorted(red),
            'blue': blue,
        }
    return out


def build_dlt(rows):
    """大乐透期号保持 5 位（与现有 dlt_data.json 约定一致）。"""
    out = {}
    for cells in rows:
        if len(cells) < 8:
            continue
        period = cells[0].strip()
        if not re.match(r'^\d{5}$', period):
            continue
        front = [int(x) for x in cells[1:6] if x.isdigit()]
        back = [int(x) for x in cells[6:8] if x.isdigit()]
        if len(front) != 5 or len(back) != 2:
            continue
        out[period] = {
            'period': period,
            'date': _first_date(cells),
            'front': sorted(front),
            'back': sorted(back),
        }
    return out


# ============ 写出 / 合并 ============

def write_file(path, schema, periods):
    periods.sort(key=lambda x: x['period'], reverse=True)
    out = {
        '#schema': schema,
        'last_updated': datetime.now(TZ).strftime('%Y-%m-%dT%H:%M:%S+08:00'),
        'total_periods': len(periods),
        'periods': periods,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    return len(periods)


def merge_existing(path, backfill):
    """以回填数据为准；保留现有文件中回填未覆盖的期号。"""
    if not os.path.exists(path):
        return backfill
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return backfill
    for p in data.get('periods', []):
        per = p.get('period')
        if per and per not in backfill:
            backfill[per] = p
    return backfill


# ============ 主流程 ============

def backfill_game(session, game, start, build_fn, out_file, schema, min_rows):
    end = os.environ.get('BF_END', '26999')
    print(f'[{game.upper()}] 拉取全量历史 (start={start}, end={end})...')
    html = fetch_full(session, game, start, end)
    rows = parse_500_rows(html)
    print(f'[{game.upper()}] 解析到 {len(rows)} 行')
    if len(rows) < min_rows:
        raise RuntimeError(
            f'[{game.upper()}] 解析行数 {len(rows)} 低于安全下限 {min_rows}，'
            f'疑似接口变动或抓取截断，已中止以免写入残缺数据。'
        )
    data = build_fn(rows)
    data = merge_existing(out_file, data)
    n = write_file(out_file, schema, list(data.values()))
    print(f'[{game.upper()}] 已写入 {n} 期 -> {os.path.relpath(out_file)}')
    return n


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    session = create_session()

    ssq_n = backfill_game(session, 'ssq', SSQ_START, build_ssq, SSQ_FILE, 'SSQ_DATA_V1', SSQ_MIN_ROWS)
    time.sleep(1.0)  # 礼貌性间隔
    dlt_n = backfill_game(session, 'dlt', DLT_START, build_dlt, DLT_FILE, 'DLT_DATA_V1', DLT_MIN_ROWS)

    print('\n========== 全量回填完成 ==========')
    print(f'双色球: {ssq_n} 期')
    print(f'大乐透: {dlt_n} 期')
    print('下一步建议: 运行 model.py 重新训练模型并生成共识，然后提交。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
