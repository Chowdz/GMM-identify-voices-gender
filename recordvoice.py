"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/10/16 14:51 
"""

import pyaudio
import wave
from tqdm import tqdm
import librosa.display
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal


def record_audio(wave_out_path, record_second):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print("* recording")
    for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        wf.writeframes(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


def play_audio(wave_path):
    CHUNK = 1024
    wf = wave.open(wave_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    datas = []
    while len(data) > 0:
        data = wf.readframes(CHUNK)
        datas.append(data)
    for d in tqdm(datas):
        stream.write(d)
    stream.stop_stream()
    stream.close()
    p.terminate()


def predict_gender(voice_path):
    y, sr = librosa.load(voice_path, sr=44100, mono=True, offset=1, duration=5)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C0'),
                                                 fmax=librosa.note_to_hz('C7'))
    f0 = [round(k, 3) for k in np.nan_to_num(f0) if k != 0]
    f0_average = round(np.average(f0), 3)
    mel = np.round(pd.DataFrame(librosa.feature.melspectrogram(y=y, sr=sr)), 3)
    mel_average = [np.average(mel.iloc[:, j]) for j in range(len(mel.columns.values))]
    IQR = round(np.percentile(mel_average, 75) - np.percentile(mel_average, 25), 3)
    print(f"你的声音基频为：{f0_average}")
    print(f"你的声音频率四分位数为：{IQR}")
    reference_data = pd.DataFrame(pd.read_csv('feature_ult.csv'))
    p_data = pd.DataFrame(pd.read_csv('p_ult.csv'))
    u_list = list(reference_data.iloc[:, 0])
    sigma_list = list(reference_data.iloc[:, 1])
    p_list = list(p_data.iloc[:, 0])
    mean1 = [u_list[0], u_list[1]]
    mean2 = [u_list[2], u_list[3]]
    cov1 = [[sigma_list[0], p_list[0] * np.power(sigma_list[0], 0.5) * np.power(sigma_list[1], 0.5)],
            [p_list[0] * np.power(sigma_list[0], 0.5) * np.power(sigma_list[1], 0.5), sigma_list[1]]]
    cov2 = [[sigma_list[2], p_list[1] * np.power(sigma_list[2], 0.5) * np.power(sigma_list[3], 0.5)],
            [p_list[1] * np.power(sigma_list[2], 0.5) * np.power(sigma_list[3], 0.5), sigma_list[3]]]
    goal_data_l = [f0_average - 0.05, IQR - 0.05]
    goal_data_r = [f0_average + 0.05, IQR + 0.05]
    pro1_l = multivariate_normal.cdf(goal_data_l, mean=mean1, cov=cov1)
    pro1_r = multivariate_normal.cdf(goal_data_r, mean=mean1, cov=cov1)
    pro2_l = multivariate_normal.cdf(goal_data_l, mean=mean2, cov=cov2)
    pro2_r = multivariate_normal.cdf(goal_data_r, mean=mean2, cov=cov2)
    pro1 = abs(pro1_r - pro1_l)
    pro2 = abs(pro2_r - pro2_l)
    pro_female = pro1 / (pro1 + pro2)
    pro_male = pro2 / (pro1 + pro2)
    if pro_female >= pro_male:
        play_audio('female_beep.wav')
        print(f"你是女生的概率为：{pro_female}")
        print(f"你是男生的概率为：{pro_male}")
        print('你是女生')
    else:
        play_audio('male_beep.wav')
        print(f"你是女生的概率为：{pro_female}")
        print(f"你是男生的概率为：{pro_male}")
        print('你是男生')
    return


if __name__ == '__main__':
    record_audio("output.wav", record_second=6)
    play_audio("output.wav")
    predict_gender('output.wav')
