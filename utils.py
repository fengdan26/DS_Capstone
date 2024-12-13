import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datetime import datetime
import os

torch.set_default_dtype(torch.float32)  # 设置默认浮点精度为 float32


def plot_candles(pricing, wins: list = [5, 20], title=None, color_function=None):
    def default_color(index, open_price, close_price, low, high):
        if open_price[index] > close_price[index]:
            return 'r'
        elif open_price[index] < close_price[index]:
            return 'g'
        else:
            return 'b'

    def volume_color_function(index, volume_change_rate):
        return 'g' if volume_change_rate[index] > 0 else 'r'

    def plot_price(ax, x, oc_max, oc_min, open_price, close_price, low, high, candle_colors, title=None):
        candles = ax.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors, align='center', width=1)
        lines1 = ax.bar(x, oc_min - low, bottom=low, color='b', linewidth=0, align='center', width=1)
        lines2 = ax.bar(x, high - oc_max, bottom=oc_max, color='b', linewidth=0, align='center', width=1)
        ax.axis("off")
        if title:
            ax.set_title(title)

    def plot_volume(ax, x, volume, volume_colors, title=None):
        candles = ax.bar(x, volume, color=volume_colors, linewidth=0, align='center', width=1)
        ax.axis('off')
        if title:
            ax.set_title(title)

    color_function = color_function or default_color
    open_price = pricing['Open'].reset_index(drop=True)
    close_price = pricing['Close'].reset_index(drop=True)
    low = pricing['Low'].reset_index(drop=True)
    high = pricing['High'].reset_index(drop=True)
    volume = pricing['Volume change rate'].reset_index(drop=True)

    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    x = list(range(len(pricing)))  # 将索引映射为整数范围

    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    volume_colors = [volume_color_function(i, volume) for i in x]

    im_arrs = []
    for i in range(len(wins)):
        fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.4), dpi=100)

        # 打印选定窗口数据
        print(f"Pricing data for window (last {wins[i]} rows):")
        print(pricing.iloc[-wins[i]:])

        # 验证颜色函数
        '''for j in range(len(candle_colors)):
            print(f"Index {j}: Open={open_price[j]}, Close={close_price[j]}, Color={candle_colors[j]}")'''
        '''for j in range(len(volume_colors)):
            print(f"Index {j}: Volume={volume[j]}, Volume Color={volume_colors[j]}")'''

        plot_price(axs[0], x[-wins[i]:], oc_max[-wins[i]:], oc_min[-wins[i]:],
                   open_price[-wins[i]:], close_price[-wins[i]:], low[-wins[i]:], high[-wins[i]:], candle_colors[-wins[i]:])
        plot_volume(axs[1], x[-wins[i]:], volume[-wins[i]:], volume_colors[-wins[i]:])

        plt.show()

        fig.tight_layout()
        fig.canvas.draw()
        im_arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        im_arr = im_arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        im_arr = im_arr[:, :, :3]

        im_arr = np.transpose(im_arr, (2, 0, 1))
        '''print(f"Image array min: {im_arr.min()}, max: {im_arr.max()}, shape: {im_arr.shape}")'''
        im_arrs.append(im_arr)
        plt.close()
    '''print(f"Raw chart RGB range: min={im_arr.min()}, max={im_arr.max()}, shape={im_arr.shape}")'''
    return im_arrs



def save_k_chart(path_symbol_lst, windows=[5, 20], target_size=(64, 64)):
    with open(path_symbol_lst) as file:
        symbol_list = file.readlines()
        for e in symbol_list[1:]:
            e = e.split(",")
            e = [s.strip() for s in e]
            symbol, index, train_episode, trajectory, step = e
            df = pd.read_csv("./SSE50_ALL/" + symbol + '.csv')
            df.rename(columns={
                "股票名称": "Name",
                "股票代码": "Symbol",
                "日期": "Date",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low",
                "成交量": "Volume",
                "成交额": "Amount",
                "振幅": "Amplitude",
                "涨跌额": "Change amount",
                "涨跌幅": "Change rate",
                "换手率": "Turnover Rate"
            }, inplace=True)
            df['Volume change rate'] = (df['Volume'] / df['Volume'].shift(1) - 1).fillna(0)
            index = int(index)
            start_index = df.index > index - max(windows)
            end_index = df.index <= index
            df = df[start_index & end_index]
            img_list = plot_candles(df, windows, target_size=target_size)
            '''for i, img in enumerate(img_list):
                print(f"Generated image {i} with shape: {img.shape}")  # 检查图像数组的形状'''


if __name__ == "__main__":
    path_symbol_lst = "./test_generate_img.txt"
    print("Starting the image generation process...")
    save_k_chart(path_symbol_lst)
    print("Image generation completed.")

