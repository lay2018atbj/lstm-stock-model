# coding:utf-8
import pandas as pd
import os
import datetime
import tushare as ts
from config import tickets, out_path, output_path, use_today, today, yesterday

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

# get the items history stock data, T+1 update
# init
history_data_path = output_path + 'history' + '.csv'
if not os.path.exists(history_data_path):
    history_data_list = []
    print('history data 初始化')
    for key, val in tickets.items():
        print(tickets[key])
        for item in tickets[key]:
            df = ts.get_k_data(code=item, start='2013-01-01')
            df['code'] = df['code'].astype(str)
            if df.shape[0] == 0:
                pass
            else:
                history_data_list.append(df)
    history_df = pd.concat(history_data_list)
else:
    history_df = pd.read_csv(history_data_path, dtype={'code': 'object'}, keep_default_na=True)
    history_df = history_df[history_df['code'].isin(list(stock_set))]

# incremental update
history_stock_set = set(history_df['code'])
stock_hist_diff_list = list(stock_set - history_stock_set)
max_date_df = history_df.groupby(['code'])['date'].max().reset_index()
max_date = yesterday
stock_date_diff_list = list(max_date_df[max_date_df['date'] != max_date]['code'])
stock_diff_list = list(set(stock_date_diff_list + stock_hist_diff_list))
history_data_list = [history_df]
i = 0
while (len(stock_diff_list) > 0 and i <= 3):
    empty_result_list = []
    print('history data 第 %s 次 循环查询' % i)
    handled_list = []
    for item in stock_diff_list:
        try:
            start_date = max_date_df.loc[max_date_df['code'] == item, 'date'].values[0]
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            start_date = start_date + datetime.timedelta(1)
            start_date = start_date.strftime('%Y-%m-%d')
        except:
            start_date = '2013-01-01'
        df = ts.get_k_data(code=item, start=start_date)
        df['code'] = df['code'].astype(str)
        if df.shape[0] == 0:
            empty_result_list.append(item)
        else:
            history_data_list.append(df)
        handled_list.append(item)

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

if not ts.is_holiday(today):
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

union_df.sort_values(by=['code', 'date'], ascending=True, inplace=True)
union_df.reset_index(drop=True, inplace=True)
union_df['index'] = union_df.index
union_yestoday_df = union_df.copy(deep=True)
union_yestoday_df['index'] = union_yestoday_df['index'] - 1
union_yestoday_df.columns = ['date_yestoday', 'trade_yestoday', 'code', 'index']
print('1:', union_yestoday_df.tail())
print('2:', union_df.tail())
union_df = union_yestoday_df.merge(union_df, on=['code', 'index'])


union_df['percent'] = (union_df['trade_yestoday'] - union_df['trade']) / union_df['trade']
print('3:', union_df.tail())

union_df = union_df.loc[:, ['date_yestoday', 'code', 'percent']]
union_df.columns = ['date', 'code', 'percent']

'''
union_df['code'] = union_df['code'].astype(str)
print(union_df.tail())
# union_df.set_index(union_df['date'], drop=True, inplace=True)
result_df = union_df.groupby(['code', 'date'])['percent'].mean().reset_index()
result_df = pd.pivot(result_df, index="date", columns="code", values="percent").reset_index()
result_df = result_df.sort_values('date', ascending=True)
result_df.to_csv(output_path + 'percent_result' + '.csv', index=False, na_rep=0)
'''

union_df = union_df.merge(tickets_df, on='code')
result_df = union_df.groupby(['block', 'date'])['percent'].mean().reset_index()
result_df = pd.pivot(result_df, index="date", columns="block", values="percent").reset_index()
result_df = result_df.sort_values('date', ascending=True)
result_df.to_csv(output_path + 'percent_result' + '.csv', index=False, na_rep=0)