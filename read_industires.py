# -*- coding: utf-8 -*-

import pandas as pd
import os

file_path = './industries/'
file_list = os.listdir(file_path)

df_all = pd.DataFrame()
for file_name in file_list:
    full_name = file_path + file_name
    df = pd.read_excel(full_name, encoding="unicode_escape", error_bad_lines=False, converters={'股票代码': str})
    df_all = pd.concat([df_all, df])

df_all.reset_index(inplace=True, drop=True)
# df_all.to_csv('all_tickets.csv')

agriculture = df_all[df_all['行业名称'] == '农林牧渔']['股票代码'].values.tolist()
excavation = df_all[df_all['行业名称'] == '采掘']['股票代码'].values.tolist()
chemical = df_all[df_all['行业名称'] == '化工']['股票代码'].values.tolist()
steel = df_all[df_all['行业名称'] == '钢铁']['股票代码'].values.tolist()
metals = df_all[df_all['行业名称'] == '有色金属']['股票代码'].values.tolist()
electronic = df_all[df_all['行业名称'] == '电子']['股票代码'].values.tolist()
electrical = df_all[df_all['行业名称'] == '电气设备']['股票代码'].values.tolist()
food = df_all[df_all['行业名称'] == '食品饮料']['股票代码'].values.tolist()
cloths = df_all[df_all['行业名称'] == '纺织服装']['股票代码'].values.tolist()
lightIndustry = df_all[df_all['行业名称'] == '轻工制造']['股票代码'].values.tolist()
medical = df_all[df_all['行业名称'] == '医药生物']['股票代码'].values.tolist()
public = df_all[df_all['行业名称'] == '公用事业']['股票代码'].values.tolist()
transport = df_all[df_all['行业名称'] == '交通运输']['股票代码'].values.tolist()
house = df_all[df_all['行业名称'] == '房地产']['股票代码'].values.tolist()
trade = df_all[df_all['行业名称'] == '商业贸易']['股票代码'].values.tolist()
service = df_all[df_all['行业名称'] == '休闲服务']['股票代码'].values.tolist()
integrated = df_all[df_all['行业名称'] == '综合']['股票代码'].values.tolist()
building = df_all[df_all['行业名称'] == '建筑材料']['股票代码'].values.tolist()
decorating = df_all[df_all['行业名称'] == '建筑装饰']['股票代码'].values.tolist()
electEquipment = df_all[df_all['行业名称'] == '家用电器']['股票代码'].values.tolist()
war = df_all[df_all['行业名称'] == '国防军工']['股票代码'].values.tolist()
computer = df_all[df_all['行业名称'] == '计算机']['股票代码'].values.tolist()
media = df_all[df_all['行业名称'] == '传媒']['股票代码'].values.tolist()
communication = df_all[df_all['行业名称'] == '通信']['股票代码'].values.tolist()
bank = df_all[df_all['行业名称'] == '银行']['股票代码'].values.tolist()
finance = df_all[df_all['行业名称'] == '非银金融']['股票代码'].values.tolist()
automobile = df_all[df_all['行业名称'] == '汽车']['股票代码'].values.tolist()
mechanics = df_all[df_all['行业名称'] == '机械设备']['股票代码'].values.tolist()

tickets = {'agriculture': agriculture, 'excavation': excavation, 'chemical': chemical, 'steel': steel,
           'metals': metals, 'electronic': electronic, 'electrical': electrical, 'food': food, 'cloths': cloths,
           'lightIndustry': lightIndustry, 'medical': medical, 'public': public, 'transport': transport, 'house': house,
           'trade': trade, 'service': service, 'integrated': integrated, 'building': building,
           'decorating': decorating, 'electEquipment': electEquipment, 'war': war, 'computer': computer,
           'media': media, 'communication': communication, 'bank': bank,
           'finance': finance, 'automobile': automobile, 'mechanics': mechanics}


