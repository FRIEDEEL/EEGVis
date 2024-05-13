# ------------------------- I am a split line \(ow o) -------------------------
from bdf_data_preprocess.process import *
from scipy.fft import fft, fftfreq, fftshift
import json
import torch
from torch.utils.data import Dataset
# ------------------------- I am a split line \(ow o) -------------------------
import os


# ------------------------- I am a split line \(ow <) -------------------------
INTERVAL=(20,120)
def main():
    # test()

    process_files_in_dir(read_dir="../data/raw/formal_50",
                         save_dir="../data/processed/6")
    # process_and_save(read_from="../data/raw/formal_50/230402_Liguang_2",save_to="../data/processed")
    pass


# ------------------------- I am a split line \(ow <) -------------------------
def process_files_in_dir(read_dir, save_dir):
    """Entrance to process all raw data under a directory.

    Each raw data should be stored under a child directory under `read_dir`, and \
    each should contain 2 files: `data.bdf` and `evt.bdf`. 

    If you want to modify the way in which the data is processed, better to \
    do this in `process_and_save()` function, and in `bdf_data_process.process.\
    DataBDF` methods.

    Args:
        - read_dir (str): the directory where data subdirs are at.
        - save_dir (str): the directory where processed `.pth` file is saved.

    Returns:
        int : 0. this return does not have any actual meaning.
    """
    subdirs = os.listdir(read_dir)
    success_count = 0
    fail_count = 0
    for subdir in subdirs:
        try:
            process_and_save(read_from=os.path.join(read_dir, subdir),
                             save_to=os.path.join(save_dir, subdir))
            success_count += 1
        except:
            print("Fail processing file : \"{}\"".format(subdir))
            fail_count += 1
        finally:
            pass
    print("Successfully processed {} files, {} failed.".format(success_count, fail_count))
    return 0


def process_and_save(read_from, save_to=None, subject=0):
    """Processes data in a single trial, and save to a `.pth` file (as tensors).


    Args:
        read_from (str): The directory where events and trial data is saved.

        save_to (str, optional): If a file is given, save data to the \
            specified file. Defaults to None.

        subject (int, optional): Specifies the subject ID. Now discarded. \
            Defaults to 0.
    """    
    data = DataBDF(filepath=read_from, subjectID=subject)
    # notch filter 参数默认
    data.apply_notch_filter(notch_freq=50.0, Q_factor=25)
    # butterworth filter 参数默认
    data.apply_butter_bandpass(order=2)
    # 数据切片
    data.slice_data()
    # 切到固定长度
    data.cut_sliced(INTERVAL)
    # 归一化数据
    data.normalize_AS()
    # 判断是否存储，没给路径就不存！
    if save_to:
        if os.path.splitext(save_to)[-1]=="pth":
            data.save_to_tensor(save_to)
        else:
            data.save_to_tensor(save_to+".pth")
    else:
        data.save_to_tensor()


# ------------------------- I am a split line \(ow o) -------------------------
def test():
    data = DataBDF(filepath='data/raw/230402_Liguang_1')
    data.apply_notch_filter(notch_freq=50.0, Q_factor=25)
    data.apply_butter_bandpass(order=2)
    data.slice_data()
    data.normalize()
    # print(np.mean(data.sliced[126]["data"][0]))
    for c in [3, 5, 13, 27]:
        for i in [2, 9, 26, 41]:
            fig = plt.figure(figsize=[10, 8])
            for chan in range(16):
                ax = fig.add_subplot(16, 1, chan + 1)
                ax.plot(data.sliced[c * 50 + i]["data"][chan])
                ax.set_ylabel("{}".format(
                    data.sliced[c * 50 + i]["info"]["chnames"][chan]))
                ax.set_ylim([-3, 3])
            plt.savefig('数据图片/class {}, sample{}.jpg'.format(
                data.sliced[c * 50 + i]["label"], i))
            # plt.show()
    # data.plot_unsliced(channels=[2,3,4,5],interval=(10000,10500),original=False)


# ------------------------- I am a split line \(ow o) -------------------------
# 查看切片后数据的样子
# fig=plt.figure()
# sliceddata=data.sliced[162]['data']
# for i in range(16):
#     ax=fig.add_subplot(16,1,i+1)
#     ax.plot(sliceddata[i,:])
#     ax.set_yticks([])
# print(data.sliced[162]['label'])
# plt.show()
# ------------------------- I am a split line \(ow o) -------------------------
# 利用FFT在频域上分析，对每个channel都做fft变换。
# fig=plt.figure()
# for i in range(16):
#     yf=fft(data.processed[i])
#     N=len(data.processed[0])
#     T=1/1000
#     xf=fftfreq(N,T)
#     xf=fftshift(xf)
#     yplot=fftshift(yf)
#     ax=fig.add_subplot(16,1,i+1)
#     ax.plot(xf, 1.0/N * np.abs(yplot))
#     ax.set_xlim(14,71)
#     ax.set_yticks([])
# plt.show()
# ------------------------- I am a split line \(ow o) -------------------------
# 利用FFT在频域上分析
# fig=plt.figure()

# yf=fft(data.processed[5])
# N=len(data.processed[0])
# T=1/1000
# xf=fftfreq(N,T)
# xf=fftshift(xf)
# yplot=fftshift(yf)
# ax=fig.add_subplot(111)
# ax.plot(xf, 1.0/N * np.abs(yplot))
# ax.set_xlim(14,71)
# plt.show()
# ------------------------- I am a split line \(ow o) -------------------------
# 尝试不同的中心频率的滤波效果
# fig=plt.figure()
# ax=[]
# for i in range(10):
#     freq=49.9+i*0.01
#     data.apply_notch_filter(notch_freq=freq)
#     ax.append(fig.add_subplot(10,1,i+1))
#     ax[i].plot(
#         data.timeseq[10000:10500],
#         data.processed[2,10000:10500]
#     )
#     ax[i].legend("{:.3f}".format(freq))
#     data.reset_processed()
#     ...
# plt.show()
# ------------------------- I am a split line \(ow <) -------------------------
# 尝试不同Q因子的滤波效果
# fig=plt.figure()
# ax=[]
# for i in range(11):
#     Qf=101-i*10
#     data.apply_notch_filter(Q_factor=Qf)
#     ax.append(fig.add_subplot(11,1,i+1))
#     ax[i].plot(
#         data.timeseq[10000:10500],
#         data.processed[2,10000:10500]
#     )
#     ax[i].legend("{:.3f}".format(Qf))
#     data.reset_processed()
#     ...
# plt.show()
# ------------------------- I am a split line \(ow o) -------------------------

if __name__ == '__main__':
    main()