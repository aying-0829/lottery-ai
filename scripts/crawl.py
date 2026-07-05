"""
AI 彩票分析工具 — 数据爬虫
爬取双色球（中彩网）和大乐透（体彩网）最新开奖数据，去重追加到 data/ 目录。
"""

import json
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
SSQ_API = 'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice'

def crawl_ssq(session):
    """
    从中彩网拉取最新双色球开奖数据。
    返回: 新增期数列表 (period strings)
    """
    print('[SSQ] 开始爬取双色球数据...')

    # 读取现有数据，获取最新期号
    existing_periods = set()
    existing_data = None
    if os.path.exists(SSQ_FILE):
        with open(SSQ_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for p in existing_data.get('periods', []):
            existing_periods.add(p['period'])
        print(f'[SSQ] 已有 {len(existing_periods)} 期数据，最新期号: {max(existing_periods) if existing_periods else "无"}')

    # 参数：拉取最近 5 期（通常 1-2 期是新数据）
    params = {
        'name': 'ssq',
        'issueCount': '5',
        'pageNo': '1',
        'pageSize': '5',
        'systemType': 'PC'
    }

    resp = session.get(SSQ_API, params=params, headers=get_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get('state') != 0 and data.get('result') is None:
        print(f'[SSQ] API 返回异常: {data}')
        return []

    records = (data.get('result') or [])

    new_periods = []
    for rec in records:
        period = rec.get('code', '')
        if period in existing_periods:
            continue

        red_str = rec.get('red', '')
        blue_str = rec.get('blue', '')

        red = [int(r) for r in red_str.split(',') if r.strip()]
        blue = int(blue_str) if blue_str else 0

        if len(red) != 6 or not (1 <= blue <= 16):
            print(f'[SSQ] 期号 {period} 数据异常: red={red}, blue={blue}，跳过')
            continue

        entry = {
            'period': period,
            'date': rec.get('date', ''),
            'red': sorted(red),
            'blue': blue
        }
        new_periods.append(entry)
        print(f'[SSQ] 新增期号: {period} date={entry["date"]} red={red} blue={blue}')

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
DLT_API = 'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry'

def crawl_dlt(session):
    """
    从体彩网拉取最新大乐透开奖数据。
    返回: 新增期数列表 (period strings)
    """
    print('[DLT] 开始爬取大乐透数据...')

    existing_periods = set()
    existing_data = None
    if os.path.exists(DLT_FILE):
        with open(DLT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for p in existing_data.get('periods', []):
            existing_periods.add(p['period'])
        print(f'[DLT] 已有 {len(existing_periods)} 期数据，最新期号: {max(existing_periods) if existing_periods else "无"}')

    params = {
        'gameNo': '85',
        'provinceId': '0',
        'pageSize': '5',
        'isVerify': '1',
        'pageNo': '1',
        'termLimits': '5'
    }

    headers = get_headers(referer='https://www.lottery.gov.cn/')
    resp = session.get(DLT_API, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get('errorCode') != '0':
        print(f'[DLT] API 返回异常: {data}')
        return []

    records = data.get('value', {}).get('list', [])

    new_periods = []
    for rec in records:
        period = rec.get('lotteryDrawNum', '')
        if period in existing_periods:
            continue

        result = rec.get('lotteryDrawResult', '')
        parts = result.strip().split()
        if len(parts) < 2:
            print(f'[DLT] 期号 {period} 开奖结果格式异常: {result}')
            continue

        front_str = parts[0]
        back_str = parts[1]

        front = [int(x) for x in front_str.split(' ') if x.strip()]
        back = [int(x) for x in back_str.split(' ') if x.strip()]

        if len(front) != 5 or len(back) != 2:
            print(f'[DLT] 期号 {period} 数据异常: front={front}, back={back}，跳过')
            continue

        entry = {
            'period': period,
            'date': rec.get('lotteryDrawTime', '')[:10],
            'front': sorted(front),
            'back': sorted(back)
        }
        new_periods.append(entry)
        print(f'[DLT] 新增期号: {period} date={entry["date"]} front={front} back={back}')

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

    # 双色球
    try:
        rate_limit()
        new = crawl_ssq(session)
        total_new['ssq'] = new
    except Exception as e:
        print(f'[SSQ] 爬取失败: {e}')
        total_new['ssq'] = []

    # 大乐透
    try:
        rate_limit()
        new = crawl_dlt(session)
        total_new['dlt'] = new
    except Exception as e:
        print(f'[DLT] 爬取失败: {e}')
        total_new['dlt'] = []

    # 汇总并写标记文件供 model.py 判断是否有新数据
    summary = {
        'ssq_new': len(total_new.get('ssq', [])),
        'dlt_new': len(total_new.get('dlt', [])),
        'timestamp': datetime.now(TZ).isoformat()
    }

    meta_file = os.path.join(DATA_DIR, '.crawl_meta.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False)

    print(f'\n========== 爬取完成 ==========')
    print(f'双色球新增: {summary["ssq_new"]} 期')
    print(f'大乐透新增: {summary["dlt_new"]} 期')
    print(f'时间戳: {summary["timestamp"]}')

    # 返回 exit code: 有新增数据 → 0，无新增 → 0（正常退出，model.py 自行判断）
    return 0


if __name__ == '__main__':
    sys.exit(main())
