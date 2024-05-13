import numpy as np
from scipy import signal
import os
import mne
import matplotlib.pyplot as plt
import copy
import sys
import json
import torch
if __name__ == '__main__':
    sys.path.append(os.getcwd())

from neuracle_lib.readbdfdata import readbdfdata

class DataBDF(object):
    def __init__(self, filepath=None,subjectID=0):
        # data是2维数组，第一维是通道
        # 第二维是数据点
        self.filename=filepath.split('/')[-1]
        self.data = None
        # events是2维数组，第一维代表不同event，
        # 第二维0代表出现时间，1代表持续时间，2代表event的类别编号
        self.events = None
        self.sfreq = 0.0  # 采样频率，float
        self.chnames = []  # 通道名称，列表list
        self.nchan = 0  # 通道数，int
        self.subjectID = subjectID

        # filepath被传入的时候初始化
        if filepath:
            self.read_data_using_neuracle_lib(filepath)
        else:
            print("Warning : The file path is not given when realizing DataBDF\
                 instance.\\ The data might not be initialized properly.")
        self._get_time_seq()
        # 一份数据的拷贝，用于储存处理过的数据，而保留原始的数据self.data
        self.processed = self.data.copy()
        # 切片后的数据，总之先定义一下。列表，每一个列表存储
        self.sliced = []

    def read_data_using_neuracle_lib(self, path):
        """Reads data from the `.bdf file`. This method is called in __init__()
        method.

        To read the events, this method calls `readbdfdata` method from
        `neuracle_lib.readbdfdata`. 

        Args:
            path (_str_): The directory of data stored. Should contain 
            `data.bdf` and `events.bdf`
        """
        # 直接用neuracle_lib.readbdfdata中方法读取event和data
        self._rawdata = readbdfdata(path)
        self.events = self._rawdata['events']  # event，2维数组
        self.data = self._rawdata['data']  # 数据，2维数组
        self.sfreq = self._rawdata['srate']  # 采样频率
        self.nchan = self._rawdata['nchan']  # 通道数量
        self.chnames = self._rawdata['ch_names']  # 通道名称（电极名称）

    # 将预处理过的data重新变为未处理过的data，撤销所有修改。
    def reset_processed(self):
        """Resets the data to unprocessed version by re-initializing
        `self.processed` with `self.data`
        """
        self.processed = self.data.copy()

    # 向外接口，用于访问EEG数据
    def get_data(self):
        ...

    # 用于生成一个时间序列。可能在做图的时候有用
    def _get_time_seq(self):
        # 简单粗暴的做法，data的随便一个通道的长度，再除去采样率。我觉得可能要改改
        self.timeseq = np.arange(len(self.data[0])) / self.sfreq

    # 用notch filter 滤波，滤去50Hz交流电噪声
    # 利用FFT分析可以找出较佳的频率和Q因子。
    def apply_notch_filter(self, notch_freq=50.0, Q_factor=25.0):
        """Apply a notch filter to data.
        
        The notch filter is from `scipy.signal`.

        Args:
            notch_freq (float, optional): The frequency of notch filter.
                Defaults to 50.0.
            Q_factor (float, optional): The quality factor. The larger value
                indicates more narrow range near the central frequency to be
                filtered out. Defaults to 25.0.
        """        
        samp_freq = self.sfreq
        b_notch, a_notch = signal.iirnotch(notch_freq, Q_factor, samp_freq)
        for chan in range(self.nchan):
            self.processed[chan] = signal.filtfilt(b_notch, a_notch,
                                                   self.processed[chan])

    def apply_butter_bandpass(self, order=2, lowcut=14, highcut=71):
        butter = signal.butter(order, [lowcut, highcut],
                               fs=self.sfreq,
                               btype='band',
                               output='sos')
        for chan in range(self.nchan):
            self.processed[chan] = signal.sosfiltfilt(butter,
                                                      self.processed[chan])

    # 把数据剪切。详见文档
    def slice_data(self, max_duration=600, discard=0, min_duration=50, label_offset=0,stop_tri=None,):
        """Slice the data. And save it in `self.sliced`.

        0504 UPDATE: Use this function just to seperate slices with different \
        events. DOES NOT cut the slice into specific length. Use `cut` instead.

        Args:
            max_duration (int, optional): The duration of image shown in *ms*. \
                Defaults to 600.

            discard (int, optional): The number of sample points to discard. \
                Defaults to 40.

            stop_tri (int, optional): If stop trigger is given to determine \
                whether a image is done showing, we slice the data in a \
                different way. Defaults to None.
        """
        # TODO 给了停止信号的时候的剪切方式
        if stop_tri:
            ...
        # 没给停止信号的时候的剪切方式
        else:
            for i in range(len(self.events)):
                tempdata=None
                time_start = self.events[i][0]
                # 设定终止位置。由于i+1可能超出索引范围导致报错，故用try语句
                try:
                    # time_int是两次event之间的间距
                    time_int = self.events[i + 1][0] - self.events[i][0]
                    if time_int<50:
                        continue
                    # 如果两个event之间相距过长(超过原来的duration 100ms)，我们认为它是一个break
                    # 这组数据的slice在区间(time_start,time_start+max_duration)
                    elif time_int > (max_duration + 100):
                        tempdata=copy.deepcopy(self.processed[:, time_start : time_start + max_duration])
                    # 除此之外的情况(event之间时间和duration差不多)，以time_end(下一个event的开始时间)为准。
                    else:
                        time_end = self.events[i + 1][0]
                        tempdata=copy.deepcopy(self.processed[:, time_start : time_end])
                # IndexError i+1超出索引范围，就是到最后一组event了
                # 处理方式和break前的那组处理方式相同
                except IndexError:
                    tempdata=copy.deepcopy(self.processed[:, time_start : time_start + max_duration])
                    break
                # 用finally语句，无论是否超出index都进行判断是否储存
                finally:
                    # 仅在标签合法(取值范围在[1,40]之间)时append数据。
                    if self.events[i][2]+label_offset-1 in range(40):
                        self.sliced.append({
                            'data': # 数据
                                tempdata,
                            'label': # 标签，由于之前实验设计的问题这里要-1。TODO:优化这部分。
                                self.events[i][2]+label_offset-1
                            })
                    else:
                        print("label {} discarded at {}".format(self.events[i][2],i))
        return self

    def cut_sliced(self,cut_interval=(40,480)):
        # 简单检查一下区间是否合法。
        if cut_interval[0]<0 or cut_interval[0]>=cut_interval[1]:
            raise ValueError("Interval not valid! Double check!")
        i=0
        while i < len(self.sliced):
            item=self.sliced[i]
            # 如果切割区间过长，抛弃这组数据
            if isinstance(item["data"],np.ndarray):
                i+=1
                if cut_interval[1]>item["data"].shape[1]:
                    item["data"]=None
                else:
                    item["data"]=item["data"][:,cut_interval[0]:cut_interval[1]]
            else:
                self.sliced.pop(i)

        return self

    # 在指定区间上作出信号图像，此处plot出来的数据还没切片
    def plot_unsliced(self,
                       channels=None,
                       interval=(10000, 11000),
                       original=False):
        """Plotting the unsliced data, just for quick preview.

        Args:
            channels (list, optional): List of channels to plot. Defaults to None.
            interval (tuple, optional): The time interval where the data is
                displayed. Defaults to (10000, 11000).
            original (bool, optional): Whether to plot the processed data or
                the unprocessed. Defaults to False.
        """
        # 未指定channel时默认对全部频道做图
        if not channels:
            channels = range(self.nchan)
        fig = plt.figure()
        axes = []
        for i in range(len(channels)):
            axes.append(fig.add_subplot(len(channels), 1, i + 1))
            if original:
                axes[i].plot(
                    self.timeseq[interval[0]:interval[1]],  # x轴为时间
                    self.data[
                        channels[i],
                        interval[0]:interval[1]],  # y轴为channels[i]通道的原始数据
                )
                axes[i].set_yticks([])
            else:
                axes[i].plot(
                    self.timeseq[interval[0]:interval[1]],  # x轴为时间
                    self.processed[
                        channels[i],
                        interval[0]:interval[1]],  # y轴为channels[i]通道的处理后数据
                )
                axes[i].set_yticks([])
        plt.show()
    
    # NOTE(B.Lee, 230404) 感觉这个函数要弃用了，没办法保存np文件，而且直接torch.save
    # 感觉更好
    def save_as_json(self, to_file=None):
        """Save the sliced data to a json file.

        Args:
            file_path (str): the path to which. If not given, save to \
            `data/processed/filename` where `filename` shared from the raw data.
            Defaults to None.
        """
        if len(self.sliced)==0:
            print("Warning: Data not sliced, writting empty content to data \
                file")
        if to_file:
            _file_path=to_file
        else:
            _file_path='data/processed/{}.json'.format(self.filename)
        print(self.filename)
        with open(_file_path,'w+') as f:
            json.dump({
                'data' : self.sliced,
                'info' : {
                    'chnames' : self.chnames,
                    'sfreq' : self.sfreq,
                }},f)

    def save_to_tensor(self, to_file=None):
        """Convert the data from `self.sliced` to `torch.Tensor` and save
        it in file.

        Args:
            to_file (str, optional): The file to which the data is saved.
            Uses `self.filename` under `data/processed` if not specified.
            Defaults to None.
            
        """        
        _data=[]
        for item in self.sliced:
            if isinstance(item["data"],np.ndarray):
                _data.append({
                    'data' : torch.from_numpy(item['data']).float(),
                    'label' : (item['label']),
                })
        if to_file:
            _file_path=to_file
        else:
            raise Exception("Error: no save path specified")
        _save={
            'data': _data,
            'info': {
                'sfreq' : self.sfreq,
                'nchan' : self.nchan,
                'chnames' : self.chnames,
            },
            'subject': self.subjectID
        }
        torch.save(_save, _file_path)

    def normalize_BS(self,):

        pass

    def normalize_AS(self,
                     mode = "standard",
                     on_allchan = True,
                     on_alltime = False):
        """Normalizes the sliced data. Still numpy based. AS stands for after slice.

        Args:
            mode (str, optional): The way of normalizing the sliced data. \
                Could be specified to following values:
                "standard": Standard deviation normalization.
                Defaults to "standard".
        """
        if len(self.sliced)==0:
            raise Exception("Data is not sliced yet. Use DataBDF.slice_data() \
                            to slice the data first")
        if mode=="standard":
            for item in self.sliced:
                data=item["data"]
                if not isinstance(item["data"],np.ndarray):
                    continue
                # print(type(data))

                else:
                    mean=np.mean(data)
                    std_dev=np.std(data)
                    item["data"]=(data-mean)/std_dev
        else:
            print("Unable to recognize the mode of normalization \"{}\"".format(mode))
        
# ------------------------- I am a split line \(ow o) -------------------------


def main():
    pass

if __name__ == '__main__':
    main()