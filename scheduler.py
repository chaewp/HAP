"""
HAP - Auto Crawler Scheduler
============================
매주 자동으로 크롤러를 실행해서 데이터를 쌓습니다.
Usage:
  python scheduler.py
"""

import schedule
import time
import subprocess
import sys
from datetime import datetime

def run_crawlers():
    print(f"\n{'='*50}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Starting weekly data collection...")
    print('='*50)

    crawlers = ['crawler.py', 'crawler_iphone.py', 'crawler_macbook.py']
    for crawler in crawlers:
        print(f"\n▶ Running {crawler}...")
        try:
            result = subprocess.run([sys.executable, crawler], capture_output=True, text=True, timeout=120)
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            if result.returncode != 0:
                print(f"  ⚠️ Error: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ {crawler} timed out.")
        except Exception as e:
            print(f"  ⚠️ {crawler} failed: {e}")

    print(f"\n✅ Data collection complete at {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 매주 월요일 오전 9시에 실행
schedule.every().monday.at("09:00").do(run_crawlers)

print("HAP Auto Scheduler started!")
print("Crawlers will run every Monday at 09:00.")
print("Press Ctrl+C to stop.\n")

# 시작하자마자 한 번 실행
run_crawlers()

while True:
    schedule.run_pending()
    time.sleep(60)
