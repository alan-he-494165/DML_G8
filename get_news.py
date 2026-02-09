# 导入tushare和pandas
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import os
import calendar
import pickle
import time
import sys
from dataclasses import dataclass

# 定义新闻数据类
@dataclass
class NewsItem:
    datetime: str
    content: str
    channels: str

pro = ts.pro_api('31c93f930abfd98726b28a7d984e5d56163f69c1f040ba97d6d4436a')

cache_folder = "news_cache"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)
    print(f"创建文件夹: {cache_folder}")

start_year = 2020
end_year = 2025

for year in range(start_year, end_year + 1):
    year_folder = os.path.join(cache_folder, str(year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    
    print(f"\n========== 正在处理 {year} 年 ==========")
    
    for month in range(1, 13):
        month_folder = os.path.join(year_folder, f"{month:02d}")
        if not os.path.exists(month_folder):
            os.makedirs(month_folder)
        
        days_in_month = calendar.monthrange(year, month)[1]
        
        print(f"\n--- {year}-{month:02d} ---")
        
        for day in range(1, days_in_month + 1):
            current_date = datetime(year, month, day)
            day_end = current_date + timedelta(days=1)
            
            day_str = current_date.strftime("%Y-%m-%d")
            day_end_str = day_end.strftime("%Y-%m-%d")
            
            print(f"正在拉取 {day_str} 的数据...", end="")
            sys.stdout.flush()
            
            # 用于存储该天的数据
            daily_news_items = []
            offset = 0
            page_size = 1500
            total_fetched = 0
            
            while True:
                try:
                    print(f" 第{offset // page_size + 1}页...", end="")
                    sys.stdout.flush()
                    
                    df = pro.news(**{
                        "start_date": day_str,
                        "end_date": day_end_str,
                        "src": "sina",
                        "limit": str(page_size),
                        "offset": str(offset)
                    }, fields=[
                        "datetime",
                        "content",
                        "channels"
                    ])
                    
                    if df is None or df.empty:
                        print(" 无更多数据", end="")
                        break
                    
                    # 保存获取的数据
                    rows_count = len(df)
                    for _, row in df.iterrows():
                        news_item = NewsItem(
                            datetime=row['datetime'],
                            content=row['content'],
                            channels=row['channels']
                        )
                        daily_news_items.append(news_item)
                    
                    total_fetched += rows_count
                    print(f"({rows_count}条,总计{total_fetched}条)", end="")
                    sys.stdout.flush()
                    
                    # 如果返回的数据少于page_size，说明已经是最后一页了
                    if rows_count < page_size:
                        print(" 已获取全部", end="")
                        break
                    
                    # 否则继续获取下一页
                    offset += page_size
                    time.sleep(0.3)  # 避免请求过快
                    
                except Exception as e:
                    print(f" 错误: {e}", end="")
                    sys.stdout.flush()
                    break
            
            # 保存该天的数据到 pickle 文件
            pickle_file = os.path.join(month_folder, f"{day_str}.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(daily_news_items, f)
            
            print(f" ✓ 共保存 {len(daily_news_items)} 条数据到 {day_str}.pkl")
            sys.stdout.flush()
            
            # 添加延迟，避免API限速
            time.sleep(0.5)

print(f"\n{'='*50}")
print(f"所有数据拉取完成！")
print(f"数据存储位置: cache/YYYY/MM/YYYY-MM-DD.pkl")
print(f"覆盖时间范围: {start_year}-01-01 到 {end_year}-12-31")
print(f"{'='*50}")