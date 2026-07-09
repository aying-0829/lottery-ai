"""
AI 彩票分析工具 — 数据爬虫
爬取双色球（中彩网）和大乐透（体彩网）最新开奖数据，去重追加到 data/ 目录。
"""

import json
import re
import os
import sys
import time
import random
from datetime import datetime, timezone, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ 配置 ============

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
UA_POOL = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
]

TZ = timezone(timedelta(hours=8))  # UTC+8

# ============ HTTP 会话 ============

def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET']
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=5)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def get_headers(referer=None):
    return {
        'User-Agent': random.choice(UA_POOL),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        **({'Referer': referer} if referer else {})
    }


def rate_limit():
    time.sleep(random.uniform(1.5, 3.0))


# ============ 双色球爬取 ============

SSQ_FILE = os.path.join(DATA_DIR, 'ssq_data.json')
# 500.com 双色球历史（服务端渲染表格 <tbody id="tdata">，海外 IP 可访问；
# 中彩网 cwl.gov.cn 对 GitHub 美国 IP 返回 403，故改用 500.com）
SSQ_API = 'https://datachart.500.com/ssq/history/history.shtml'


def parse_500_rows(html):
    """从 500.com history.shtml 的 <tbody id="tdata"> 提取每行单元格文本列表。"""
    m = re.search(r'<tbody id="tdata">(.*?)</tbody>', html, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    rows = []
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.DOTALL):
        tr = re.sub(r'<!--.*?-->', '', tr, flags=re.DOTALL)  # 去掉注释里伪装成的假 <td>
        cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)
        cleaned = [c.replace('&nbsp;', '').strip() for c in cells]
        if cleaned:
            rows.append(cleaned)
    return rows


def crawl_ssq(session):
    """
    从 500.com 拉取最新双色球开奖数据（服务端渲染表格，海外 IP 可访问）。
    返回: 新增期数列表 (period strings)
    """
    print('[SSQ] 开始爬取双色球数据（数据源: 500.com）...')

    existing_periods = set()
    existing_data = None
    if os.path.exists(SSQ_FILE):
        with open(SSQ_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for p in existing_data.get('periods', []):
            existing_periods.add(p['period'])
        print(f'[SSQ] 已有 {len(existing_periods)} 期数据，最新期号: {max(existing_periods) if existing_periods else "无"}')

    # 500.com 双色球期号为 7 位（如 2026077），用 7 位范围覆盖整个 2026 年
    params = {'start': '2026001', 'end': '2026200'}
    headers = get_headers(referer='https://datachart.500.com/ssq/')
    resp = session.get(SSQ_API, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = 'gb2312'
    rows = parse_500_rows(resp.text)

    new_periods = []
    for cells in rows:
        if len(cells) < 9:
            continue
        period = cells[0]
        # 500.com 期号为 5 位（如 26077），统一转为 7 位标准格式（2026077）以兼容现有数据
        if len(period) == 5:
            period = '20' + period
        if not re.match(r'^\d{7}$', period):
            continue
        if period in existing_periods:
            continue
        red = [int(x) for x in cells[1:7] if x.isdigit()]
        blue = int(cells[7]) if cells[7].isdigit() else 0
        if len(red) != 6 or not (1 <= blue <= 16):
            print(f'[SSQ] 期号 {period} 数据异常: red={red}, blue={blue}，跳过')
            continue
        date_str = cells[-1] if re.match(r'\d{4}-\d{2}-\d{2}', cells[-1]) else ''
        entry = {
            'period': period,
            'date': date_str,
            'red': sorted(red),
            'blue': blue
        }
        new_periods.append(entry)
        print(f'[SSQ] 新增期号: {period} date={date_str} red={red} blue={blue}')

    if new_periods:
        all_periods = (existing_data['periods'] if existing_data else []) + new_periods
        all_periods.sort(key=lambda x: x['period'], reverse=True)
        output = {
            '#schema': 'SSQ_DATA_V1',
            'last_updated': datetime.now(TZ).strftime('%Y-%m-%dT%H:%M:%S+08:00'),
            'total_periods': len(all_periods),
            'periods': all_periods
        }
        with open(SSQ_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)
        print(f'[SSQ] 已写入 {len(new_periods)} 条新数据，总计 {len(all_periods)} 期')
    else:
        print('[SSQ] 无新数据')

    return [p['period'] for p in new_periods]


# ============ 大乐透爬取 ============

DLT_FILE = os.path.join(DATA_DIR, 'dlt_data.json')
# 500.com 大乐透历史（服务端渲染表格；newinc/history.php 为 JS 动态加载，curl 取不到，必须用 history.shtml 传统版）
DLT_API = 'https://datachart.500.com/dlt/history/history.shtml'

def crawl_dlt(session):
    """
    从 500.com 拉取最新大乐透开奖数据（服务端渲染表格，海外 IP 可访问）。
    返回: 新增期数列表 (period strings)
    """
    print('[DLT] 开始爬取大乐透数据（数据源: 500.com）...')

    existing_periods = set()
    existing_data = None
    if os.path.exists(DLT_FILE):
        with open(DLT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for p in existing_data.get('periods', []):
            existing_periods.add(p['period'])
        print(f'[DLT] 已有 {len(existing_periods)} 期数据，最新期号: {max(existing_periods) if existing_periods else "无"}')

    params = {'start': '26001', 'end': '26100'}
    headers = get_headers(referer='https://datachart.500.com/dlt/')
    resp = session.get(DLT_API, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = 'gb2312'
    rows = parse_500_rows(resp.text)

    new_periods = []
    for cells in rows:
        if len(cells) < 8:
            continue
        period = cells[0]
        if not re.match(r'^\d{5}$', period):
            continue
        if period in existing_periods:
            continue
        front = [int(x) for x in cells[1:6] if x.isdigit()]
        back = [int(x) for x in cells[6:8] if x.isdigit()]
        if len(front) != 5 or len(back) != 2:
            print(f'[DLT] 期号 {period} 数据异常: front={front}, back={back}，跳过')
            continue
        date_str = cells[-1] if re.match(r'\d{4}-\d{2}-\d{2}', cells[-1]) else ''
        entry = {
            'period': period,
            'date': date_str,
            'front': sorted(front),
            'back': sorted(back)
        }
        new_periods.append(entry)
        print(f'[DLT] 新增期号: {period} date={date_str} front={front} back={back}')

    if new_periods:
        all_periods = (existing_data['periods'] if existing_data else []) + new_periods
        all_periods.sort(key=lambda x: x['period'], reverse=True)
        output = {
            '#schema': 'DLT_DATA_V1',
            'last_updated': datetime.now(TZ).strftime('%Y-%m-%dT%H:%M:%S+08:00'),
            'total_periods': len(all_periods),
            'periods': all_periods
        }
        with open(DLT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)
        print(f'[DLT] 已写入 {len(new_periods)} 条新数据，总计 {len(all_periods)} 期')
    else:
        print('[DLT] 无新数据')

    return [p['period'] for p in new_periods]


# ============ 主流程 ============

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    session = create_session()

    total_new = {}
    failed = {}

    # 双色球
    try:
        rate_limit()
        new = crawl_ssq(session)
        total_new['ssq'] = new
        failed['ssq'] = False
    except Exception as e:
        print(f'[SSQ] 爬取失败: {e}')
        total_new['ssq'] = []
        failed['ssq'] = True

    # 大乐透
    try:
        rate_limit()
        new = crawl_dlt(session)
        total_new['dlt'] = new
        failed['dlt'] = False
    except Exception as e:
        print(f'[DLT] 爬取失败: {e}')
        total_new['dlt'] = []
        failed['dlt'] = True

    # 汇总并写标记文件供 model.py / CI 判断是否有新数据、是否失败
    summary = {
        'ssq_new': len(total_new.get('ssq', [])),
        'dlt_new': len(total_new.get('dlt', [])),
        'ssq_failed': failed.get('ssq', False),
        'dlt_failed': failed.get('dlt', False),
        'timestamp': datetime.now(TZ).isoformat()
    }

    meta_file = os.path.join(DATA_DIR, '.crawl_meta.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False)

    print(f'\n========== 爬取完成 ==========')
    print(f'双色球新增: {summary["ssq_new"]} 期' + (' [失败]' if summary['ssq_failed'] else ''))
    print(f'大乐透新增: {summary["dlt_new"]} 期' + (' [失败]' if summary['dlt_failed'] else ''))
    print(f'时间戳: {summary["timestamp"]}')

    # 返回 exit code: 脚本已正常结束，单档失败的信息记录在 .crawl_meta.json 的
    # ssq_failed / dlt_failed 字段中。是否判定为"致命失败"交给调用方（CI）根据
    # 这两个字段决定——这里统一返回 0，避免某一档被海外 IP 封禁就拖垮整个部署
    # （例如双色球的中彩网对 GitHub 美国 IP 返回 403，但大乐透仍可正常更新）。
    return 0


if __name__ == '__main__':
    sys.exit(main())
