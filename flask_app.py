# coding:utf-8
# @Time    : 2020/4/11 1:06 下午
# @Author  : qihong.fu
# @File    : flask_app.py
# @Software: PyCharm
from flask import *
import os


app = Flask(__name__)
app.config['email']='123456@gmail.com'#账号密码
app.config['password']='123456'
app.secret_key=os.urandom(24)



@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'GET':
        if request.args.get('email') != app.config['email']:
            error = 'Invalid username'
        elif request.args.get('password') != app.config['password']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for("stock"))
    return render_template('index.html', error=error)


@app.route('/stock', methods=['GET', 'POST'])
def stock():
    tickets_list = ['agriculture_excavation_chemical_steel',
                    'bank_finance_automobile_mechanics',
                    'cloths_lightIndustry_medical_public',
                    'integrated_building_decorating_electEquipment',
                    'metals_electronic_electrical_food',
                    'transport_house_trade_service',
                    'war_computer_media_communication']

    return render_template('export.html', tickets_list = tickets_list)


if __name__ == "__main__":
    app.run()
