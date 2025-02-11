import cdsapi
import pandas
import calendar
# 从 2004 年 1 月 1 日开始下载
# 每次下载一个月，存为一个文件

start_time = '2004-01-01'
end_time = '2024-03-01'

client = cdsapi.Client()

def get_days(year, month):
    return [str(i).zfill(2) for i in range(1, calendar.monthrange(year, month)[1] + 1)]

print(get_days(2004, 1))

for date in pandas.date_range(start_time, end_time, freq='M'):
    year = date.year
    month = date.month
    dataset = "satellite-sea-surface-temperature"
    request = {
        "variable": "all",
        "processinglevel": "level_4",
        "sensor_on_satellite": "combined_product",
        "version": "2_1",
        "year": [str(year)],
        "month": [str(month).zfill(2)],
        # 获取当月所有天
        "day": get_days(year, month)
    }

    client.retrieve(dataset, request).download()