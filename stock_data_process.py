# coding:utf-8
import pandas as pd
import os
import datetime
import tushare as ts
from config import tickets, out_path, output_path, use_today, today, yesterday
import time

import json
import requests
import time

# mkdir output file
if not os.path.exists(out_path):
    os.mkdir(out_path)

# get the stock block config dataframe
tickets_df_list = []
for key, val in tickets.items():
    df_temp = pd.DataFrame({'block': key, 'code': val})
    tickets_df_list.append(df_temp)
tickets_df = pd.concat(tickets_df_list)
tickets_df['code'] = tickets_df['code'].astype(str)
stock_set = set(tickets_df['code'])
print(stock_set)
print(len(stock_set))

# get the items history stock data, T+1 update
# init
data_dir = 'data/'
history_data_path = output_path + 'history' + '.csv'
if not os.path.exists(history_data_path):
    history_data_list = []
    print('history data 初始化')
    for key, val in tickets.items():
        print(tickets[key])
        for item in tickets[key]:
            print('start download:', item)
            try:
                if os.path.exists(data_dir + item + '.csv'):
                    df = pd.read_csv(data_dir + item + '.csv',
                                     dtype={'code': 'str'}, keep_default_na=True)
                else:
                    df = ts.get_k_data(code=item, start='2013-01-01')
                    df['code'] = df['code'].astype(str)
                if df.shape[0] == 0:
                    pass
                else:
                    df.to_csv(data_dir + item + '.csv', index=False)
                    history_data_list.append(df)
            except:
                print('falied download:', item)
                pass
    history_df = pd.concat(history_data_list)
    history_df.to_csv(history_data_path, index=False)
else:

    history_df = pd.read_csv(history_data_path,
                             dtype={'code': 'str'}, keep_default_na=True)

    print(history_df.head(5))
    print('stock_set:', stock_set)
    print('history_set', set(history_df['code']))

    history_df = history_df[history_df['code'].isin(list(stock_set))]
    print(len(set(history_df['code'])))

# incremental update
history_stock_set = set(history_df['code'])
stock_hist_diff_list = list(stock_set - history_stock_set)
print(len(stock_hist_diff_list))
max_date_df = history_df.groupby(['code'])['date'].max().reset_index()
max_date = '2020-04-30'

# 此函数不准
'''
while ts.is_holiday(max_date):
    max_date_d = datetime.datetime.strptime(max_date, '%Y-%m-%d') + datetime.timedelta(-1)
    max_date = max_date_d.strftime('%Y-%m-%d')

print('max_date after:', max_date)
'''
stock_date_diff_list = list(max_date_df[max_date_df['date'] != max_date]['code'])
stock_diff_list = list(set(stock_date_diff_list + stock_hist_diff_list))
history_data_list = [history_df]
i = 0

print('diff len=', len(stock_diff_list))

while (len(stock_diff_list) > 0 and i < 0):
    empty_result_list = []
    print('history data 第 %s 次 循环查询' % i)
    handled_list = []
    for item in stock_diff_list:
        print('try download:', item)
        try:
            start_date = max_date_df.loc[max_date_df['code'] == item, 'date'].values[0]
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            start_date = start_date + datetime.timedelta(1)
            start_date = start_date.strftime('%Y-%m-%d')
        except:
            start_date = '2013-01-01'
        print(item)
        df = ts.get_k_data(code=item, start=start_date)
        # df = pd.DataFrame()
        # print(df.empty)
        if not df.empty:
            df['code'] = df['code'].astype(str)
            df.to_csv(data_dir + item + '.csv', index=False)
        else:
            print('download failed:', item)
        if df.shape[0] == 0:
            empty_result_list.append(item)
        else:
            history_data_list.append(df)
        handled_list.append(item)

    print('empty_result_list lenL', len(empty_result_list))
    history_df = pd.concat(history_data_list)
    history_data_list = [history_df]
    history_stock_set = set(history_df['code'])
    stock_hist_diff_list = list(stock_set - history_stock_set)
    max_date_df = history_df.groupby(['code'])['date'].max().reset_index()
    max_date = max_date_df['date'].max()
    stock_date_diff_list = list(max_date_df[max_date_df['date'] != max_date]['code'])
    stock_diff_list = list(set(stock_date_diff_list + stock_hist_diff_list)
                           - set(empty_result_list))
    print(stock_diff_list)
    i = i + 1

history_df = history_df.drop_duplicates()
history_df = history_df[history_df['code'].isin(list(stock_set))]
history_df.to_csv(output_path + 'history' + '.csv', index=False)

history_df = history_df[['date', 'close', 'code']]
union_df = history_df.rename(columns={'close': 'trade'})
# get today items stock data
today_data_path = output_path + 'today' + '.csv'

if use_today and not ts.is_holiday(today):
    if os.path.isfile(output_path + today + '.csv'):
        today_df = pd.read_csv(output_path + today + '.csv')
    else:
        today_df = ts.get_today_all()
        today_df.to_csv(output_path + today + '.csv')
    today_df['code'] = today_df['code'].astype(str)
    today_df['date'] = today
    today_df.to_csv(today_data_path, index=False)
    today_df = today_df[['date', 'trade', 'code']]

    # union today and history df
    today_df = today_df[today_df['code'].isin(list(set(history_df['code'])))]
    union_df = pd.concat([union_df, today_df], sort=False)
    union_df = union_df.drop_duplicates()



# stock block data
union_df_agg = union_df.groupby(['code'])['trade'].agg({'min': 'min', 'max': 'max'}).reset_index()
union_df = union_df.merge(union_df_agg, on='code')

union_df['trade'] = (union_df['trade'] - union_df['min']) / (union_df['max'] - union_df['min'])
union_df['code'] = union_df['code'].astype(str)
union_df = union_df.merge(tickets_df, on='code')

result_df = union_df.groupby(['block', 'date'])['trade'].mean().reset_index()

result_df = pd.pivot(result_df, index="date", columns="block", values="trade").reset_index()
result_df = result_df.sort_values('date', ascending=True)
result_df.to_csv(output_path + 'result' + '.csv', index=False, na_rep=0)
