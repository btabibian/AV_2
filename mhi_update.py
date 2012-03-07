import urllib2
import sys
import time
from math import cos, sin
import cv
class mhi():
    def __init__(self):
        self.CLOCKS_PER_SEC = 1
        self.MHI_DURATION = 5

        self.MAX_TIME_DELTA = 0.5
        self.MIN_TIME_DELTA = 0.05
        self.N = 100

        self.buf = range(100) 
        self.last = 0
        self.mhi = None # MHI
        self.orient = None # orientation
        self.mask = None # valid orientation mask
        self.segmask = None # motion segmentation map
        self.storage = None # temporary storage
        self.counter=0
    def update_mhi(self,img, dst, diff_threshold):
        last=self.last
        mhi=self.mhi
        storage=self.storage
        mask=self.mask
        orient=self.orient
        segmask=self.segmask
        counter=self.counter
        buf=self.buf
        N=self.N
        MIN_TIME_DELTA=self.MIN_TIME_DELTA
        MAX_TIME_DELTA = self.MAX_TIME_DELTA
        CLOCKS_PER_SEC=self.CLOCKS_PER_SEC
        MHI_DURATION =self.MHI_DURATION

        timestamp = counter#time.clock() / CLOCKS_PER_SEC # get current time in seconds
        counter=counter+0.5
        size = cv.GetSize(img) # get current frame size
        idx1 = last
        if not mhi or cv.GetSize(mhi) != size:
            for i in range(N):
                buf[i] = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
                cv.Zero(buf[i])
            mhi = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
            cv.Zero(mhi) # clear MHI at the beginning
            orient = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
            segmask = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
            mask = cv.CreateImage(size,cv. IPL_DEPTH_8U, 1)
    
        cv.CvtColor(img, buf[last], cv.CV_BGR2GRAY) # convert frame to grayscale
        idx2 = (last + 1) % N # index of (last - (N-1))th frame
        last = idx2
        silh = buf[idx2]
        cv.AbsDiff(buf[idx1], buf[idx2], silh) # get difference between frames
    
        cv.Threshold(silh, silh, diff_threshold, 1, cv.CV_THRESH_BINARY) # and threshold it
        cv.UpdateMotionHistory(silh, mhi, timestamp, MHI_DURATION) # update MHI
        cv.CvtScale(mhi, mask, 255./MHI_DURATION,
                    (MHI_DURATION - timestamp)*255./MHI_DURATION)
        cv.Zero(dst)
        cv.Merge(mask, None, None, None, dst)
        cv.CalcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3)
        if not storage:
            storage = cv.CreateMemStorage(0)
        seq = cv.SegmentMotion(mhi, segmask, storage, timestamp, MAX_TIME_DELTA)
        for (area, value, comp_rect) in seq:
            if comp_rect[2] + comp_rect[3] > 100: # reject very small components
                color = cv.CV_RGB(255, 0,0)
                silh_roi = cv.GetSubRect(silh, comp_rect)
                mhi_roi = cv.GetSubRect(mhi, comp_rect)
                orient_roi = cv.GetSubRect(orient, comp_rect)
                mask_roi = cv.GetSubRect(mask, comp_rect)
                angle = 360 - cv.CalcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)

                count = cv.Norm(silh_roi, None, cv.CV_L1, None) # calculate number of points within silhouette ROI
                if count < (comp_rect[2] * comp_rect[3] * 0.05):
                    continue

                magnitude = 30.
                center = ((comp_rect[0] + comp_rect[2] / 2), (comp_rect[1] + comp_rect[3] / 2))
                cv.Circle(dst, center, cv.Round(magnitude*1.2), color, 3, cv.CV_AA, 0)
                cv.Line(dst,
                        center,
                        (cv.Round(center[0] + magnitude * cos(angle * cv.CV_PI / 180)),
                         cv.Round(center[1] - magnitude * sin(angle * cv.CV_PI / 180))),
                        color,
                        3,
                        cv.CV_AA,
                        0)
        self.last=last
        self.mhi=mhi
        self.storage=storage
        self.mask=mask
        self.orient=orient
        self.segmask=segmask
        self.counter=counter
        self.buf=buf
