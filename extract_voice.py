"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/10/12 20:22 
"""

import os
import csv
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alive_progress import alive_bar
import random


# 输入文件地址，返回其目录下所有语音文件地址列表
def getFileName(file_path):
    file_list = []
    f_list = os.listdir(file_path)
    for i in f_list:
        wave_path = file_path + '\\' + i
        file_list.append(wave_path)
    return file_list


# 提取基频和IQR特征，实时写入到一个文件
def voice_feature(path_list):
    with open("transient1.csv", "w", encoding="gbk", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["F0", "IQR"])
        random.seed(1)
        path_list_random = random.sample(path_list, 2000)
        with alive_bar(2000, force_tty=True) as bar:
            for i in range(2000):
                y, sr = librosa.load(path_list_random[i], sr=44100, mono=True, offset=1, duration=2)
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C0'),
                                                             fmax=librosa.note_to_hz('C7'))
                f0 = [round(k, 3) for k in np.nan_to_num(f0) if k != 0]
                f0_average = round(np.average(f0), 3)
                mel = np.round(pd.DataFrame(librosa.feature.melspectrogram(y=y, sr=sr)), 3)
                mel_average = [np.average(mel.iloc[:, j]) for j in range(len(mel.columns.values))]
                IQR = round(np.percentile(mel_average, 75) - np.percentile(mel_average, 25), 3)
                csv_writer.writerow([f0_average, IQR])
                f.flush()
                print(f"F0：{f0_average}，IQR:{IQR}")
                f0.clear()
                mel_average.clear()
                bar()
        f.close()
    return


def spec_show(path_list):
    value, catch = librosa.load(path_list, sr=44100)
    A = librosa.stft(value)
    Adb = librosa.amplitude_to_db(abs(A))
    plt.figure(figsize=(14, 11))
    librosa.display.specshow(Adb, sr=catch, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar()
    plt.show()
    return


if __name__ == '__main__':
    wave_file_path = 'E:\\Study\\Training_Dataset\\voice_chinese\\ST-CMDS-20170001_1-OS'
    wave_path_list = getFileName(wave_file_path)
    voice_feature(wave_path_list)
    # spec_show('20170001P00001A0001.wav')
