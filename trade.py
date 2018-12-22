import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import datetime
import scipy.optimize as sco
import math
import re

'''
函数执行顺序：
init(先订阅所有的期货信息) ->
before_trading(进行阿尔法策略筛选，选出20个最优的) ->
handle_bar(每天)
'''
# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def now_trade_future(context):
    '''
    获得所有品种当前主力合约的名字，只选择最活跃的
    除去CY，因为服务器数据不全
    未筛选的总信息：所有的主力合约
    '''
    total = all_instruments(type='Future').order_book_id.values
    category = set()
    pattern = re.compile(r'\D+')
    res = []
    for item in total:
        category.add(pattern.findall(item)[0])
    for name in list(category):
        if not name=='CY':
            domain = get_dominant_future(name)
            if domain:
                res.append(domain)
    return res


def init(context):
    '''
    init函数：程序运行时执行的第一个函数
    context: 全局变量 持仓、账户余额和总额和获取的期货信息均在context变量中
    把所有的期货信息复制到context.s1
    '''
    # context内引入全局变量s1
    # 初始化时订阅合约行情。订阅之后的合约行情会在handle_bar中进行更新。
    # 实时打印日志
    name = now_trade_future(context)
    context.s1 = [v for v in name if not v.endswith('88') and not v.endswith('99')]
    subscribe(context.s1)
    context.fired = False
    logger.info("RunInfo: {}".format(context.run_info))


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def alpha_select(context):
    '''
    获得期货品种的价格计算收益率，并减去无风险收益率
    获得沪深300的价格计算收益率，并减去无风险收益率
    对沪深300和期货品种进行拟合，取收益率最高的
    '''
    now_hold = list(context.future_account.positions.keys())
    res = []
    res.extend(list(set(now_hold) & set(context.s1)))
    num_sure = len(res)
    ranking = []
    top_rank = 20 - num_sure
    compute_future = list(set(context.s1) - set(now_hold))
    for i in compute_future:
        risk_free = context.risk_free
        try:
            future_price = pd.DataFrame(
                get_price(i, start_date=context.now - datetime.timedelta(days=context.look_back),
                          frequency='1d', fields='close'))
            lm_data = pd.merge(context.index, future_price, left_index=True, right_index=True, how='inner')
            index_ret = price2return(lm_data.close_x)
            future_ret = price2return(lm_data.close_y)
            lm_fit = np.polyfit(index_ret - risk_free, future_ret - risk_free, 1)
            ranking.append([i, lm_fit[1]])
        except:
            continue
    ranking.sort(key=lambda x: x[1], reverse=True)
    for j in range(len(ranking)):
        res.append(ranking[j][0])
        if j == top_rank - 1:
            break
    return res


def before_trading(context):
    '''
    订阅所有期货信息，然后获得无风险利率
    然后进行alpha策略
    详见alpha_select函数
    '''
    name = now_trade_future(context)
    context.s1 = [v for v in name if not v.endswith('88') and not v.endswith('99')]
    context.look_back = 41
    subscribe(context.s1)
    context.risk_free = get_yield_curve()['0S'].values
    context.index = pd.DataFrame(
        get_price('沪深300', start_date=context.now - datetime.timedelta(days=context.look_back)).close)
    context.after_alpha = alpha_select(context)
    context.operations = [[], []]
    context.down_check = True


def price2return(minute_price):
    '''
    获得收益率
    '''
    log_Return = np.array([])
    for i in range(len(minute_price)):
        if i != 0:
            log_Return = np.concatenate(
                (log_Return, np.array([np.log(minute_price[i] / minute_price[i - 1])])))
    return log_Return


def Each_futures(choice, log_Return):
    '''
    获得d阶差分，d取决于AD检验，检测单位根是否平稳
    通过acf分析自相关性，通过Ljung-Box Q-Statistic求出自相关性的阶数(95%置信的自相关系数不为0)
    pacf求出偏相关性的阶数(90%置信的偏相关性不为0)
    
    '''
    def index_process(series):
        res = [series[0]];
        error_num = 0
        for i in range(1, len(series)):
            if series[i] - series[i - 1] >= 3:
                return res
            elif series[i] - series[i - 1] == 2:
                error_num += 1
                if error_num >= 2:
                    return res
                else:
                    res.append(series[i])
            elif series[i] - series[i - 1] == 1:
                res.append(series[i])
        return res

    def select_process(series):
        new_series = index_process(series)
        if len(new_series) <= 2:
            return new_series
        else:
            return new_series[-2:]

    try:
        d = 0
        for i in range(5):
            ts_test = np.diff(log_Return, i)
            p = adfuller(ts_test, 1)[1]
            if p > 0.05:
                d = i
                break
                # Get d

        p_series = acf(np.diff(log_Return, d), nlags=5, qstat=True)[2]
        acf_index = [0]
        for i in range(len(p_series)):
            if p_series[i] > 0.05:
                acf_index.append(i + 1)
        acf_index = select_process(acf_index)
        # print('acf_index=', acf_index)
        # Get p

        q_series = pacf(np.diff(log_Return, d), nlags=5)
        pacf_index = []
        for i in range(len(q_series)):
            if abs(q_series[i]) > 0.1:
                pacf_index.append(i)
        pacf_index = select_process(pacf_index)
        # print('pacf_index=', pacf_index)
        # Get q

        selected_tuple = []
        for i in pacf_index:
            for j in acf_index:
                selected_tuple.append((i, d, j))
        selected_tuple.sort(key=lambda x: sum(x))
        # Get all of the possible (p,i,q)

        max_len = min(3, len(selected_tuple))
        selected_tuple = selected_tuple[:max_len]
        # Choose top 3 or less

        dict = {}
        feasible = []
        for i in range(len(selected_tuple)):
            try:
                model = ARIMA(log_Return, order=selected_tuple[i]).fit()
                # print('selected_tuple=', selected_tuple[i])
                # print('fit succeeded')
                dict[selected_tuple[i]] = model
                ele = list(selected_tuple[i])
                a = model.aic
                ele.append(a)
                feasible.append(ele)
            except:
                continue

        feasible.sort(key=lambda x: x[3])

        final_model = dict[tuple(feasible[0][:3])]

        prediction = final_model.predict()
        return np.sum(prediction[0:choice - 1])

    except:
        return


def future_selection(context):
    '''
    获得1000分钟的回报率
    '''
    now_hold = list(context.future_account.positions.keys())
    might_be_future = []
    for i in context.after_alpha:
        try:
            Time = np.array([])
            record_price = history_bars(i, bar_count=1000, frequency='1m', fields='close')
            record_ret = price2return(record_price)

            if len(record_ret) == 999:
                sumed_log_return = Each_futures(60, record_ret)
                if sumed_log_return != None:
                    might_be_future.append([i, sumed_log_return, record_ret])
        except:
            continue

    might_be_future.sort(key=lambda x: x[1], reverse=True)
    res = []
    j=0
    while True:
        if might_be_future[j][0] in now_hold:
            res.append(might_be_future.pop(j))
        else:
            j += 1
        if j == len(might_be_future):
            break
    top_rank = 10 - len(res)
    for i in range(len(might_be_future)):
        if not i == top_rank:
            res.append(might_be_future[i])
        else:
            break
    return res


def opti_pf(res, context):  # res 是 might_be_future
    '''
    先用tangency portfolio计算出选出期货的权重
    (最优化：夏普比率)
    然后用总市值的0.85倍乘以权重计算出手数
    对手数取整(向下取整)
    '''
    try:
        number_of_assets = len(res)
        # 生成随机数
        weights = np.random.random(number_of_assets)
        # 将随机数归一化，每一份就是权重，权重之和为1
        weights /= np.sum(weights)
        pred_ret = []
        compute_cov = []
        for i in range(len(res)):
            pred_ret.append(res[i][1])
            compute_cov.append(res[i][2])
        compute_cov = np.matrix(compute_cov)

        def statistics(weights):
            weights = np.array(weights)
            pret = np.dot(np.array(pred_ret), weights) * 1000
            pvol = np.sqrt(np.dot(weights, np.dot(np.cov(compute_cov) * 1000, weights)))
            return pret / pvol

        def min_func_sharpe(weights):
            return -statistics(weights)

        bnds = tuple((0, 1) for x in range(number_of_assets))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        opts = sco.minimize(min_func_sharpe, number_of_assets * [1. / number_of_assets, ], method='SLSQP', bounds=bnds,
                            constraints=cons)

        pool = context.future_account.total_value * 0.85
        cost = opts['x'].round(6) * pool
        futures = []
        for i in range(len(res)):
            unit = current_snapshot(res[i][0]).last
            if math.isnan(unit) or unit == 0:
                unit = res[i][2][-1]
            price = unit * instruments(res[i][0]).contract_multiplier * instruments(res[i][0]).margin_rate
            if not (price == 0):
                num = math.floor(cost[i] / price)
                if num >= 1:
                    futures.append([res[i][0], num])
        return futures
    except:
        return


# 得到现在要的操作
def waiting_list(futures, context):
    if futures:
        buy_list = []
        sell_list = []
        name_list = [future[0] for future in futures]
        for item in context.future_account.positions.keys():
            if item not in name_list:
                sell_list.append([item, context.future_account.positions[item].buy_quantity])
        for future in futures:
            if future[0] not in context.future_account.positions.keys():
                buy_list.append(future)
            else:
                if future[1] > context.future_account.positions[future[0]].buy_quantity:
                    buy_list.append([future[0], future[1] - context.future_account.positions[future[0]].buy_quantity])
                if future[1] < context.future_account.positions[future[0]].buy_quantity:
                    sell_list.append([future[0], context.future_account.positions[future[0]].buy_quantity - future[1]])
        return [sell_list, buy_list]
    else:
        return [[], []]


def opera_process(context, bar_dict):
    '''
    买卖操作，先比较两个列表
    '''
    sell_list = context.operations[0]
    buy_list = context.operations[1]
    indi = 0
    while True:
        if indi == len(sell_list):
            break
        sell_list[indi][1] = min([sell_list[indi][1], context.future_account.positions[sell_list[indi][0]].buy_quantity])
        if sell_list[indi][1] == 0:
            del sell_list[indi]
            continue
        temp = bar_dict[sell_list[indi][0]].volume
        if not math.isnan(temp):
            temp = int(temp * 0.25)
            if temp > 0:
                if temp < sell_list[indi][1]:
                    sell_close(sell_list[indi][0], amount=temp)
                    sell_list[indi][1] += -temp
                if temp > sell_list[indi][1]:
                    sell_close(sell_list[indi][0], amount=sell_list[indi][1]) # sell+ LimitPrice()
                    del sell_list[indi]
                    indi = indi - 1
        indi = indi + 1
    indi = 0
    while True:
        if indi == len(buy_list):
            break
        temp = bar_dict[buy_list[indi][0]].volume
        if not math.isnan(temp):
            temp = int(temp * 0.25)
            if temp > 0:
                if temp < buy_list[indi][1]:
                    buy_open(buy_list[indi][0], amount=temp) # Change: + LimitOrder(price)
                    buy_list[indi][1] += -temp
                if temp > buy_list[indi][1]:
                    buy_open(buy_list[indi][0], amount=buy_list[indi][1])
                    del buy_list[indi]
                    indi += -1
        indi += 1
    return [sell_list, buy_list]


# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):
    '''
    定时操作 ARIMA和优化和买卖操作分别属于不同的函数
    附加：
    每20分钟查看一次，如果某一品种亏损大于10%，则全部售出
    '''
    if (context.now.date().isoweekday() in [1, 2, 3, 4, 5]) and (
        (context.now.time().hour == 10 and context.now.time().minute == 0) or (
            context.now.time().hour == 14 and context.now.time().minute == 0)):
        # 开始编写你的主要的算法逻辑
        selects = future_selection(context)
        futures = opti_pf(selects, context)
        context.operations = waiting_list(futures, context)
        # context.future_account 可以获取到当前投资组合信息
        # 使用buy_open(id_or_ins, amount)方法进行买入开仓操作

    if not context.operations == [[], []]:
        context.operations = opera_process(context, bar_dict)
    
    if (context.now.time().minute in [0,20,40]):
        now_hold = list(context.future_account.positions.keys())
        for future in now_hold:
            buy_cost = context.future_account.positions[future].buy_avg_open_price * context.future_account.positions[future].buy_quantity
            if not buy_cost==0.0:
                if context.future_account.positions[future].pnl/buy_cost < -0.1:
                    context.operations[0].append([future, context.future_account.positions[future].buy_quantity])


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
