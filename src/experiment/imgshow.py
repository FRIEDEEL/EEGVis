# ------------------------- I am a split line \(ow o) -------------------------
# import psychopy modules
from psychopy import core, visual, data, event
from psychopy.tools.filetools import fromFile, toFile
from psychopy.preferences import prefs
from psychopy.core import quit
# ------------------------- I am a split line \(ow o) -------------------------
# import modules of neuracle
if __name__=="__main__":
    import neuracle_lib
else:
    import sys
    import os
    sys.path.append(os.getcwd())
import neuracle_lib
from neuracle_lib.triggerBox import TriggerIn
from neuracle_lib.triggerBox import PackageSensorPara
# ------------------------- I am a split line \(ow o) -------------------------
# ordinary imports
import numpy
import random
import os
import sys
# ------------------------- I am a split line \(ow o) -------------------------

class ImageLib(object):
    ...

class SingleClassLib(ImageLib):
    """ImageLib containing images from a single class.

    """    
    def __init__(self,image_dir: str,classID:int=None):
        """Ini the object

        Args:
            image_dir (str): The path that contains multiple images.
            classID (int, optional): The ID of class of images in. Reads from 
                `info.txt` if not specified. Defaults to None.
        """        
        self.image_dir=image_dir # Directory of image to display
        self.files=os.listdir(self.image_dir)
        if classID:
            self.classID=int(classID)
        else:
            with open(os.path.join(image_dir,'info.txt')) as f:
                self.classID=int(f.readline().split(',')[1])
        self.files.sort()
        for file in self.files:
            if ".jpeg" in file:
                ...
            elif ".jpg" in file:
                ...
            elif ".png" in file:
                ...
            else:
                self.files.remove(file)
                
    def getid(self,index:int):
        '''Get the ID of class of ImageLib object'''
        return self.classID

    def gen_list_of_paths():
        pass

    def getimage(self, index:int):
        '''Returns the path to a image file with index given.'''
        image=os.path.join(self.image_dir,self.files[index]) # compose absolute path
        return image

class Trail():
    """A trail that displays image on screen, and send trigger to device.

    Parameters
    ----------
    window : visual.Window
        Window object of psychopy.visual
    img_lib : _ImageLib_
        ImageLib is a self-defined class. make sure the object has 2 methods:
            **getimage(self)**, returns the path of an image,
            and **getid(self)**, returns the class id of the image.
    number : _int_
        Numbers of images to be shown in the target directory(folder).
    clock : _Psychopy.core.Clock_, optional.
        This is shit, by default None
    time_display : intorfloat, optional.
        The time of each image displayed, in seconds, by default 0.5
    send_trigger : _bool_, optional.
        Whether send trigger to device.
        Often set to `False` when deploying test without connecting the device.
        by default True
    serial_name : _str_, optional
        The name of serial port. Check in terminal with command `ls /dev/cu.usbserial-*`(on MacOS),
        by default "/dev/cu.usbserial-DM6G75XB"

    Methods
    ----------
    start() : starts the trail.
    """        
    def __init__(self,
            window: visual.Window,
            img_lib: ImageLib,
            number: int = 10,
            clock=None,
            time_display: int or float = 0.5, # in seconds
            time_blank: int=0,
            send_trigger=True,
            serial_name="/dev/cu.usbserial-DM6G75XB"
            ):
    
        self.win=window # set the win where the trail is shown
        self.imglib=img_lib
        self.number=number # the number of images to be played
        self.clock=clock
        self.time_display=time_display # Time of each image to be displayed in ms
        self.time_blank=time_blank # Time of displaying a blank (usually "+" against background) between different images
        self.send_trigger=send_trigger
        if self.send_trigger:
            self.triggerin = TriggerIn(serial_name) # serial port sending triggers to device
            flag = self.triggerin.validate_device() # validates the device
    
    def start(self):
        """Starts the trail once this method is called.
        """        
        if self.clock==None:
            self.clock=core.Clock()
        start_time=self.clock.getTime()
        if self.send_trigger:
            # range for idx to traverse, update when non-image file is given
            
            for idx in range(self.number):
                # display image in display time
                stim=visual.ImageStim(self.win, image=self.imglib.getimage(idx)) # changes the image to be displayed
                stim.draw()
                self.triggerin.output_event_data(self.imglib.getid(idx)) # sent classID to device
                self.win.flip()
                # display image between time interval [(display+blank)*i,(display+blank)*i+display)
                while (self.time_display+self.time_blank)*idx<=self.clock.getTime()<(self.time_display+self.time_blank)*idx+self.time_display:
                    ...
                self.win.flip()
                # display blank frame in blank time
                # TODO: display blank frame
        #         while (self.time_display+self.time_blank)*idx<=self.clock.getTime()<(self.time_display+self.time_blank)*(idx+1):
        #             ...
        #         self.win.flip()
        else:
            for idx in range(self.number):
                # display image in display time
                try:
                    stim=visual.ImageStim(self.win, image=self.imglib.getimage(idx)) # changes the image to be displayed
                except OSError():
                    continue
                stim.draw()
                self.win.flip()
                while (self.time_display+self.time_blank)*idx<=self.clock.getTime()<(self.time_display+self.time_blank)*idx+self.time_display: # display image between time interval [(display+blank)*i,(display+blank)*i+display)
                    ...
                self.win.flip()
                # display blank frame in blank time
                # TODO: display blank frame
                # while (self.time_display+self.time_blank)*idx<=self.clock.getTime()<(self.time_display+self.time_blank)*(idx+1):
                #     ...
                # self.win.flip()

def main():
    pass

if __name__=="__main__":
    main()

# ------------------------- I am a split line \(ow o) -------------------------