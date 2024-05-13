# ------------------------- I am a split line \(ow o) -------------------------
from img_stimuli_lib.imgshow import *
# ------------------------- I am a split line \(ow o) -------------------------
import os

# 使用的图片类的父目录路径
IMG_ROOT_DIR='../../3_code/DAT_ImageNet_Images/formal_40_class'
# 是否发送Trigger信号在这里设置！
# SEND_TRIGGER=True
SEND_TRIGGER=False

def main():
    # trail_test_imagenet_10_class_50()
    trail_formal_imagenet_40_class_50()


def trail_formal_imagenet_40_class_50():
    global IMG_ROOT_DIR
    rootdir=IMG_ROOT_DIR
    imglibs=[]
    paths=os.listdir(rootdir)
    for path in paths:
        if path[0]=='n':
            imglibs.append(SingleClassLib(os.path.join(rootdir,path)))
    imglibs.sort(key=lambda lib:lib.getid()) # 按classID排序类

    # 用with语句打开Window，确保完成后关闭
    with visual.Window(
            monitor="testMonitor",
            units="deg",
            color=[-1,-1,-1], # 黑色背景
            fullscr=True # 全屏显示
            ) as win:
        # 最开始显示空白帧3秒
        blank=visual.TextStim(win,text="+")
        blank.size=10
        blank.draw()
        win.flip()
        core.wait(3)

        # 开始实验，
        for idx in range(40):
            imglib=imglibs[idx]
            trail=Trail(win,
                img_lib=imglib,
                number=50,
                time_display=0.5,
                send_trigger=SEND_TRIGGER)
            trail.start()
            blank.draw()
            win.flip()
            # 一轮结束，等5秒开始下一轮实验
            core.wait(5)

def trail_test_imagenet_10_class_50():
    imglibs=[]
    paths=os.listdir('DAT_ImageNet_Images/test_subset')
    # 确保文件夹顺序
    paths.sort()
    # 初始化本次实验的ImageLib类
    idnum=1 # id number，send给device的标签
    for path in paths:
        if path[0]=='n': # ImageNet的一类图片的文件夹开头都是'n'
            imglibs.append(SingleClassLib('DAT_ImageNet_Images/test_subset'+'/'+path, idnum))
            idnum+=1
    # 用with语句打开Window，确保完成后关闭
    with visual.Window(
            monitor="testMonitor",
            units="deg",
            color=[-1,-1,-1], # 黑色背景
            fullscr=True # 全屏显示
            ) as win:
        # 最开始显示空白帧3秒
        blank=visual.TextStim(win,text="+")
        blank.size=10
        blank.draw()
        win.flip()
        core.wait(3)

        # 开始实验，
        for imglib in imglibs:
            trail=Trail(win,imglib,50,send_trigger=SEND_TRIGGER)
            trail.start()
            blank.draw()
            win.flip()
            # 一轮结束，等5秒开始下一轮实验
            core.wait(5)



def test_trail():
    imglib=ImageLib('../ImageNet_Images/n02352591', 1)
    with visual.Window(monitor="testMonitor", units="deg",fullscr=True) as win:
        testtrail=Trail(win,imglib,50,send_trigger=True)
        testtrail.show()

# def trail_imagenet_40class_50():
#     # initialize imagelibs
#     imglib_list=[]
#     path=os.listdir('../ImageNet_Images')

    with visual.Window(monitor="testMonitor", units="deg",fullscr=True) as win:
        ...

if __name__=="__main__":
    main()