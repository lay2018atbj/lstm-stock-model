# coding:utf-8
import pandas as pd
import os

import tushare as ts
from config import tickets, out_path, output_path, use_today, today

# mkdir output file
if not os.path.exists(out_path):
    os.mkdir(out_path)

tickets_df_list = []
for key, val in tickets.items():
    df_temp = pd.DataFrame({'block': key, 'code': val})
    tickets_df_list.append(df_temp)
tickets_df = pd.concat(tickets_df_list)

# get the items history stock data, T+1 update
history_data_path = output_path + 'history-' + today + '.csv'
if not os.path.exists(history_data_path):
    history_data_list = []
    for key, val in tickets.items():
        print(tickets[key])
        for item in tickets[key]:
            df = ts.get_k_data(code=item, start='2013-01-01')
            df['code'] = df['code'].astype(str)
            history_data_list.append(df)
    history_df = pd.concat(history_data_list)
    history_df.to_csv(history_data_path, index = False)
else:
    history_df = pd.read_csv(history_data_path, keep_default_na=True)



history_df = history_df[['date', 'close', 'code']]
union_df = history_df.rename(columns = {'close':'trade'})
# get today items stock data
today_data_path = output_path + 'today-' + today + '.csv'

if use_today:
    if os.path.exists(today_data_path):
        today_df = pd.read_csv(today_data_path, keep_default_na=True)
    else:
        today_df = ts.get_today_all()
        today_df['code'] = today_df['code'].astype(str)
        today_df['date'] = today
        today_df.to_csv(today_data_path, index = False)
    today_df = today_df[['date', 'trade', 'code']]

    # check today data integrity
    set_diff = set(history_df['code']) - set(today_df['code'])
    if len(set_diff) > 0:
        print(set_diff)
        quit(-1)

    # union today and history df
    today_df = today_df[today_df['code'].isin(list(set(history_df['code'])))]
    union_df = pd.concat([union_df, today_df], sort=False)

# stock block data
union_df_agg = union_df.groupby(['code'])['trade'].agg({'min': 'min', 'max':'max'}).reset_index()
union_df = union_df.merge(union_df_agg, on='code')

union_df['trade'] = (union_df['trade'] - union_df['min'])/(union_df['max'] - union_df['min'])
union_df = union_df.merge(tickets_df, on='code')

result_df = union_df.groupby(['block', 'date'])['trade'].mean().reset_index()

result_df = pd.pivot(result_df, index="date", columns="block", values="trade").reset_index()
result_df = result_df.sort_values('date', ascending=True)
result_df.to_csv(output_path + 'result-' + today + '.csv', index = False)