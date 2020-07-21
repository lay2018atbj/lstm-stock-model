#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import requests
import time

# 获取当前时间 格式20190225
nowTime = time.strftime('%Y%m%d', time.localtime())
date = nowTime
# 节假日接口(工作日对应结果为 0, 休息日对应结果为 1, 节假日对应的结果为 2 )
server_url = "http://api.goseek.cn/Tools/holiday?date="

req = requests.get(server_url + date)

# 获取data值
vop_data = json.loads(req.text)
print('日期 ' + str(date) + '\n查询结果为 ' + str(vop_data) + '\n结论 ', end=' ')
if vop_data["data"] == 0:
    print('Its weekday')
elif vop_data["data"] == 1:
    print('Its weekend')
elif vop_data["data"] == 2:
    print('Its holiday')
else:
    print('Error')
