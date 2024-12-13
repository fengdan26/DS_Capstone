import os
import glob
import pickle
import datetime
import pandas as pd
import torch
import tushare as ts
import numpy as np
from utils import plot_candles  # 假设 plot_candles 用于生成 RGB-stick 图像

torch.set_default_dtype(torch.float32)  # 设置默认浮点精度为 float32


class Market:
    def __init__(self, path, filename, period):
        self.path = path
        self.period = period
        self.filename = filename

        # 检查 SSE50.csv 是否存在
        sse50_file = f"{self.path}/{self.filename}"
        # print(sse50_file)
        if not os.path.exists(sse50_file):
            raise FileNotFoundError(f"{self.filename} not found in the specified path: {self.path}")

        # 加载股票代码
        self.symbol_list = pd.read_csv(sse50_file, encoding='GBK')
        self.symbol_list['Symbol'] = self.symbol_list['代码'].apply(str)  # 股票代码列
        self.symbol_list = self.symbol_list.drop_duplicates(subset='Symbol') 
        self.start = period[0]
        self.end = period[1]

    def load_symbol_data(self, symbol):
        """
        加载包含指定股票代码的文件，并进行预处理。
        """
        file_pattern = os.path.join(self.path, f"*{symbol}*.csv")
        file_matches = glob.glob(file_pattern)

        if not file_matches:
            print(f"File for symbol {symbol} not found.")
            return None

        file_path = file_matches[0]
        df = pd.read_csv(file_path, encoding='GBK')
        df.rename(columns={
            "股票名称": "Name",
            "股票代码": "Symbol",
            "日期": "Date",
            "开盘价(元)": "Open",
            "收盘价(元)": "Close",
            "最高价(元)": "High",
            "最低价(元)": "Low",
            "成交量(股)": "Volume",
            "成交额": "Amount",
            "振幅": "Amplitude",
            "涨跌额": "ChangeAmount",
            "涨跌幅": "ChangeRate",
            "换手率": "TurnoverRate"
        }, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # 确保 Volume change rate 列存在
        if 'Volume' in df.columns:
            df['Volume change rate'] = (df['Volume'] / df['Volume'].shift(1) - 1).fillna(0)
        else:
            print(f"Warning: 'Volume' column not found in {symbol}, setting 'Volume change rate' to 0.")
            df['Volume change rate'] = 0

        return df

    def get_chart(self, df, start_index, end_index,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        生成特定范围的数据对应的图表，并降低分辨率（支持 GPU 加速）。
        """
        sub_df = df.iloc[start_index:end_index]       
        required_columns = ["Open", "Close", "High", "Low", "Volume change rate"]
        for col in required_columns:
            if col not in sub_df.columns:
                raise KeyError(f"Required column '{col}' is missing in the data.")

        # 生成 RGB-stick 图像
        charts = plot_candles(sub_df, wins=[end_index - start_index])
        if not charts or len(charts) == 0:
            print(f"plot_candles returned no charts for indices {start_index}-{end_index}")
            return []
        
        charts = plot_candles(sub_df, wins=[end_index - start_index])
        for i, chart in enumerate(charts):
            unique_values = np.unique(chart)
            print(f"Chart {i} unique values: {unique_values}, Shape: {chart.shape}")

        processed_charts = []
        for i, chart in enumerate(charts):
            # Debug raw chart stats
            print(f"Chart {i} before normalization: Max={chart.max()}, Min={chart.min()}, Shape={chart.shape}")
            
            # New normalization logic
            chart_min = chart.min()
            chart_max = chart.max()
            if chart_max - chart_min > 0:  # Avoid division by zero
                chart = (chart[:, :, :3] - chart_min) / (chart_max - chart_min)
            else:
                print(f"Chart {i} is uniform. Skipping normalization.")
                chart = chart[:, :, :3]  # Pass-through without normalization

            print(f"Chart {i} after normalization: Max={chart.max()}, Min={chart.min()}")

            # Convert to PyTorch tensor and resize
            tensor_chart = torch.nn.functional.interpolate(
                torch.from_numpy(chart).permute(2, 0, 1).unsqueeze(0).to(device),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            processed_charts.append(tensor_chart)

        return processed_charts

    def generate_dataset(self, n_days, x_days, output_dir):
        """
        按股票代码生成包含 condition 和 target 的数据集，使用 GPU 加速生成。
        """
        os.makedirs(output_dir, exist_ok=True)

        # 使用 GPU 设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for symbol in self.symbol_list['Symbol']:
            print(f"Processing symbol: {symbol}")
            df = self.load_symbol_data(symbol)
            if df is None or len(df) < n_days + x_days:
                print(f"Skipping symbol {symbol} due to insufficient data.")
                continue

            df = df[(df['Date'].dt.date >= self.start) & (df['Date'].dt.date <= self.end)]
            if len(df) < n_days + x_days:
                print(f"Skipping symbol {symbol} due to filtered insufficient data.")
                continue

            dataset = []

            for i in range(len(df) - n_days - x_days + 1):
                condition_indices = range(i, i + n_days)
                target_indices = range(i + n_days, i + n_days + x_days)

                # 在 GPU 上生成图像
                condition_charts = self.get_chart(df, condition_indices.start, condition_indices.stop, device=device)
                target_charts = self.get_chart(df, target_indices.start, target_indices.stop, device=device)

                # 确保条件的维度为 (n_days, channels, img_size, img_size)
                condition_charts = torch.stack(condition_charts, dim=0)  # (n_days, channels, img_size, img_size)

                # 确保目标的维度为 (x_days, channels, img_size, img_size)
                target_charts = torch.stack(target_charts, dim=0)  # (x_days, channels, img_size, img_size)

                dataset.append({
                    "condition": condition_charts.cpu().numpy(),
                    "target": target_charts.cpu().numpy(),
                })

            print(f"Generated {len(dataset)} samples for symbol {symbol}")

            # 保存为未压缩的 pickle 文件
            output_file = os.path.join(output_dir, f"{symbol}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump({"images": dataset}, f)
            print(f"Dataset for symbol {symbol} saved to {output_file}")


if __name__ == "__main__":
    # 配置路径
    data_path = r"Ashare_former_right/Ashare/"
    period = [datetime.date(2020, 1, 1), datetime.date(2022, 12, 31)]
    n_days = 20  # 用于 condition 的天数
    x_days = 20  # 用于 target 的天数
    output_dir = "hs300_stock_pickle_3y"

    def select_hs300(start_date, end_date):
        pro = ts.pro_api()
        hs300 = pro.index_weight(index_code='000300.SH', start_date=start_date, end_date=end_date)
        return hs300['con_code'].to_list()

    hs300 = select_hs300('20200101','20200102')
    # print(hs300)

    for filename in os.listdir(data_path):
        if filename[:-4] in hs300:
            # if os.path.exists(f"{output_dir}/{filename[:-4]}.pkl"):
            #     print(f"{filename[:-4]}.pkl already exists!")
            # else:
            market = Market(data_path, filename, period)
            market.generate_dataset(n_days, x_days, output_dir)

