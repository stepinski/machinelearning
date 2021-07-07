import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import time
import ruptures as rpt
from datetime import datetime

data = pd.read_csv("test_anom2.csv")
print(data.head())
data.set_index(np.arange(len(data.index)), inplace=True)


def check_if_shift_v0(data, column_name, start_index, end_index, check_period):
    """ using median to see if it changes significantly in shift """
    period_before = data[column_name][start_index - check_period: start_index]
    period_in_the_middle = data[column_name][start_index:end_index]
    period_after = data[column_name][end_index: end_index + check_period]

    period_before_median = abs(np.nanmedian(period_before))
    period_in_the_middle_median = abs(np.nanmedian(period_in_the_middle))
    period_after_median = abs(np.nanmedian(period_after))

    upper_threshold = period_in_the_middle_median * 2
    down_threshold = period_in_the_middle_median / 2

    if (upper_threshold < period_before_median and upper_threshold < period_after_median) or\
            (down_threshold > period_before_median and down_threshold > period_after_median):
        return True
    else:
        return False


def prepare_data_to_test(data, data_name: str):
    """ datetime type """
    data["time"] = pd.to_datetime(data.time)
    """ sort values """
    data.sort_values(by=['time'], inplace=True)
    """ set index """
    data.set_index("time", inplace=True)
    """ drop duplicate time"""
    data = data[~data.index.duplicated(keep='first')]
    """ resample """
    data = data.resample('5T').pad()
    """ reset index """
    data.reset_index("time", inplace=True)
    """ rename column names """
    data.columns = ['time', data_name]
    data.drop_duplicates(subset="time", inplace=True)
    return data


def prepare_data_dp(data, column_to_fix):
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = pd.to_datetime(data.time, unit='m').dt.strftime('%H:%M')
    daily_pattern = data[column_to_fix].groupby(by=[data.time.map(lambda x: (x.hour, x.minute))]).mean()
    daily_pattern = daily_pattern.reset_index()
    daily_pattern['hour'] = pd.date_range('00:00:00', periods=len(daily_pattern), freq='5min')
    daily_pattern['hour'] = pd.to_datetime(daily_pattern.hour, unit='m').dt.strftime('%H:%M')
    daily_pattern = daily_pattern[['hour', column_to_fix]]
    data['dp'] = data['hour']
    mapping = dict(daily_pattern[['hour', column_to_fix]].values)
    final_ = dict(data[['dp', column_to_fix]].values)
    z = {**final_, **mapping}
    data.index = np.arange(len(data))
    data['daily_pattern_flow'] = data['dp'].map(z)
    return data


class DpMissingValuesV0:
    def __init__(self, df_data, fit_data):
        self.data = df_data
        self.fit_period = fit_data

    @staticmethod
    def prepare_data_dp(data, column_to_fix):
        data['time'] = pd.to_datetime(data['time'])
        data['hour'] = pd.to_datetime(data.time, unit='m').dt.strftime('%H:%M')
        daily_pattern = data[column_to_fix].groupby(by=[data.time.map(lambda x: (x.hour, x.minute))]).mean()
        daily_pattern = daily_pattern.reset_index()
        daily_pattern['hour'] = pd.date_range('00:00:00', periods=len(daily_pattern), freq='5min')
        daily_pattern['hour'] = pd.to_datetime(daily_pattern.hour, unit='m').dt.strftime('%H:%M')
        daily_pattern = daily_pattern[['hour', column_to_fix]]
        data['dp'] = data['hour']
        mapping = dict(daily_pattern[['hour', column_to_fix]].values)
        final_ = dict(data[['dp', column_to_fix]].values)
        z = {**final_, **mapping}
        data.index = np.arange(len(data))
        data['daily_pattern_flow'] = data['dp'].map(z)
        return data

    @staticmethod
    def fill_missing_values(data, column_to_fix):
        data = data.copy()
        data['hour'] = pd.to_datetime(data.time, unit='m').dt.strftime('%H:%M')
        data['time'] = pd.to_datetime(data['time'])

        data_for_dp_fit = data[:10000]
        dp = DpMissingValuesV0.prepare_data_dp(data_for_dp_fit, column_to_fix)
        i = 0
        displacement = 10000
        fit_time = 0
        data['if_fixed_with_dp'] = 0
        data['is_na'] = False
        data['is_na'] = data[column_to_fix].isnull()
        to_fix = data[data['is_na'] == True]
        while i < len(to_fix):

            current_index = to_fix.index[i]
            dp_value = dp['daily_pattern_flow'][dp['hour'] == data['hour'].loc[current_index]].iloc[0]

            if data['is_na'][current_index]:
                data[column_to_fix].loc[current_index] = dp_value
                data['if_fixed_with_dp'].loc[current_index] = 1
                i += 1
                fit_time += 1

            else:
                if fit_time > 10000:
                    data_for_dp_fit = data.loc[(current_index - displacement + 1):current_index]
                    dp = DpMissingValuesV0.prepare_data_dp(data_for_dp_fit, column_to_fix)
                    fit_time = 0
                i += 1
        return data[column_to_fix]


def dp_outlier_detection_v1(data, column_name, anomaly_column_name):
    flag_name = anomaly_column_name
    data[flag_name] = 0
    """ check how it works """
    data[flag_name][data[column_name] == 0] = 1
    data['temporary_channel'] = DpMissingValuesV0.fill_missing_values(data, column_name)
    data['hour'] = pd.to_datetime(data.time, unit='m').dt.strftime('%H:%M')
    data['time'] = pd.to_datetime(data['time'])
    displacement = 10000
    start = 0
    end = 10000
    data['dp'] = 0
    data['std'] = 0
    for i in range((math.ceil(len(data) / displacement))):
        # print(start, end)

        data_for_dp_fit = data[start:end]
        daily_pattern = prepare_data_dp(data_for_dp_fit, 'temporary_channel')
        std = np.std(data_for_dp_fit['temporary_channel'].to_numpy())

        data['dp'][start:end] = daily_pattern['daily_pattern_flow']
        data['std'][start:end] = std

        data[flag_name][start:end] = data[start:end].apply(
            lambda x: 1 if x['temporary_channel'] >= (x['dp'] + x['std'] + x['std']) else 0, axis=1)
        data[flag_name][start:end] = data[start:end].apply(
            lambda x: 1 if x['temporary_channel'] <= (x['dp'] - x['std'] - x['std']) else 0, axis=1)

        data['temporary_channel'][start:end] = data[start:end].apply(
            lambda x: x['dp'] if x['temporary_channel'] >= (x['dp'] + x['std'] + x['std'])
            else x['temporary_channel'], axis=1)
        data['temporary_channel'][start:end] = data[start:end].apply(
            lambda x: x['dp'] if x['temporary_channel'] <= (x['dp'] - x['std'] - x['std'])
            else x['temporary_channel'], axis=1)

        start += displacement
        end += displacement
    return data


class WindowModelV2:
    def __init__(self, data: pd.DataFrame, column_name: str, flag_name: str, penalty: float, window: int,
                 order: int, jump: int, cost: str):
        self.window = window
        self.order = order
        self.jump = jump
        self.cost = cost
        self.clf = rpt.Window(width=window, model=cost, jump=jump)
        self.missing_data_model = DpMissingValuesV0
        self.data = data
        self.penalty = penalty
        self.column_name = column_name
        self.flag_name = flag_name

    def prepare_data(self):
        self.data['temporary_channel'] = self.missing_data_model.fill_missing_values(self.data, self.column_name)

    def predict_changepoints(self):
        self.prepare_data()

        self.data[self.flag_name][self.data[self.column_name] == 0] = 1
        signal = self.data['temporary_channel'].to_numpy()
        outliers = []
        signal_sample = signal
        now = datetime.now()
        print("fitting and predicting at =", now)
        algo = self.clf.fit(signal_sample)
        results_ = algo.predict(pen=self.penalty)
        now = datetime.now()
        print("finished fitting and predicting at =", now)
        results = np.add(results_, 0).tolist()
        outliers.extend(results)
        start = 0
        end = 1
        now = datetime.now()
        print("checking shifts at =", now)

        for _ in range(math.floor(len(outliers) / 2)):
            start_index = outliers[start]
            end_index = outliers[end]
            # check if it is shift or only two outliers
            check_if_shift = check_if_shift_v0(self.data, self.column_name, start_index, end_index, 288)
            if check_if_shift:
                self.data[self.flag_name][start_index:(end_index + 1)] = 1

            else:
                self.data[self.flag_name][start_index: start_index+1] = 1
                self.data[self.flag_name][end_index: end_index+1] = 1
            start += 2
            end += 2
        now = datetime.now()
        print("finishing checking shifts at =", now)
        return self.data


channel_to_fix = "velocity"
anomaly_column_name = "anomalies"
output_file_name = "QAQCvelocity"

""" prepare data """
data = prepare_data_to_test(data, channel_to_fix)

data = dp_outlier_detection_v1(data, channel_to_fix, anomaly_column_name)

""" fit and predict data """
model = WindowModelV2(data, "velocity", anomaly_column_name, 0.1, 20, 7, 8, "l2")
data = model.predict_changepoints()

data = data[['time', 'velocity',anomaly_column_name]]
data.columns = ['time', 'value','flag']

data.set_index("time", inplace=True)

print(data.tail())
data.to_csv(output_file_name + ".csv")