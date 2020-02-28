# lstm-stock-model
## Overview

stock model for daily interval input 

## Installation Dependencies:

* tensorflow  1.10.0

* keras   2.0.8 

* numpy   1.14.3 

* pandas  0.23.3

* tushare 1.2.21

   

## How to Run?

### 1、git clone

```shell
git clone https://github.com/lay2018atbj/lstm-stock-model.git
cd lstm-stock-model   
```

    ### 2、two ways

```shell
# step 1 get stock prices 
python get_stock.py
# step 2 feature processing
python read_stock.py
# step 3 model train and predict, lstm_28 for 28 tickets groups 
python lstm_28.py  
```

```shell
python stock_data_process.py
python lstm_28_1.py  
```



