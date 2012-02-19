import cv
import os
import fnmatch
import time
import av_utils
import mhi_update

def mouse_call_back(event,x,y,flags,param):
    global img_sub
    if(flags&cv.CV_EVENT_FLAG_LBUTTON):
        av_utils.av_debug('colors:'+str(img_sub[y,x]))#+', colors 2:'+img_sub[x][y][1]+', colors 3:'+img_sub[x][y][0])
def getBiggestCountour(Seq):
    max_area =0
    best_area=Seq
    seq=Seq
    while seq:
        area=cv.ContourArea(seq)
        if (area>max_area):
            max_area=area
            best_area=seq
        seq=seq.h_next()
    return best_area,max_area

def getHuMoments(Seq,binary=1):
    moments=cv.Moments(Seq,binary)
    return cv.GetHuMoments(moments)

Thresh	=3


path = './data/train/3-6/'
bg= './data/background1.jpg'
type=3
dirList=os.listdir(path)
cv.NamedWindow('image')
cv.SetMouseCallback('image',mouse_call_back)
av_utils.av_debug('background image loaded from: '+bg)
bg_img=cv.LoadImage(bg)
img_size=cv.GetSize(bg_img)
img_depth=bg_img.depth
img_channel=bg_img.channels
av_utils.av_debug('image info, size:'+str(img_size)+', depth:'+str(img_depth)+', channel:'+str(img_channel))
bg_gray_img=cv.CreateImage(img_size,img_depth,1)
    
cv.ShowImage('image',bg_img)
cv.WaitKey(-1)

storage=cv.CreateMemStorage(0)
bounding_rects=list()
for fname in dirList:
    if not fnmatch.fnmatch(fname,'*.jpg'):
        continue
    fname=path+fname
    img=cv.LoadImage(fname)
    img_sub=cv.CreateImage(img_size,img_depth,img_channel)
    cv.Sub(img,bg_img,img_sub)
    cv.Smooth(img_sub,img_sub,cv.CV_MEDIAN,5)
    
    img_bw_sub=cv.CreateImage(img_size,cv.IPL_DEPTH_8U,1)
    cv.InRangeS(img_sub,(0,0,27),(100,100,140),img_bw_sub)
    
    #cv.Threshold(img_gray,,Thresh,Thresh,cv.CV_THRESH_BINARY)
    #
    #img_bw_sub_eroded=cv.CreateImage(img_size,cv.IPL_DEPTH_8U,1)
    kernel=cv.CreateStructuringElementEx(7,7,3,3,cv.CV_SHAPE_CROSS)
    #cv.Erode(img_bw_sub,img_bw_sub_eroded,kernel)

    img_bw_sub_dilated=cv.CreateImage(img_size,cv.IPL_DEPTH_8U,1)
    ##kernel=cv.CreateStructuringElementEx(5,5,2,2,cv.MORPH_CROSS)
    cv.Dilate(img_bw_sub,img_bw_sub_dilated,kernel)
    ###
    img_bw_sub_dilated_preContours=cv.CreateImage(img_size,cv.IPL_DEPTH_8U,1)
    cv.Copy(img_bw_sub_dilated,img_bw_sub_dilated_preContours)
    seq=cv.FindContours(img_bw_sub_dilated_preContours,storage,cv.RETR_EXTERNAL)
    
    if(len(seq)!=0):
        cnt_biggest,area= getBiggestCountour(seq)
        if(area>13000):
            av_utils.av_debug('contours area:'+str(area))
            rect=cv.BoundingRect(cnt_biggest)
            bounding_rects.append(rect)
            cv.DrawContours(img_sub,cnt_biggest,cv.RGB(255,255,255),cv.RGB(255,255,255),0)
        ##
            av_utils.av_debug('Rectangle: '+str(rect))
        #cv.Rectangle(bg_img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),cv.RGB(0,0,0))
    cv.ShowImage('image',img_sub)
    cv.ShowImage('image_temp',img_bw_sub)
    cv.WaitKey(-1)    

largest_bound=(bounding_rects[0][0],bounding_rects[0][1],bounding_rects[0][0]+bounding_rects[0][2],bounding_rects[0][1]+bounding_rects[0][3])
for i in bounding_rects:
    if(i[0]<largest_bound[0]):
        largest_bound=(i[0],largest_bound[1],largest_bound[2],largest_bound[3])
    if(i[1]<largest_bound[1]):
        largest_bound=(largest_bound[0],i[1],largest_bound[2],largest_bound[3])
    if((i[2]+i[0])>largest_bound[2]):
        largest_bound=(largest_bound[0],largest_bound[1],(i[0]+i[2]),largest_bound[3])
    if((i[3]+i[1])>largest_bound[3]):
        largest_bound=(largest_bound[0],largest_bound[1],largest_bound[2],(i[1]+i[3]))
#cv.Rectangle(bg_img,(largest_bound[0],largest_bound[1]),(largest_bound[0]+largest_bound[2],largest_bound[1]+largest_bound[3]),cv.RGB(0,0,0))
#cv.ShowImage('image',bg_img)
#cv.WaitKey(-1)


bounding_box=(largest_bound[0],largest_bound[1],largest_bound[2]-largest_bound[0],largest_bound[3]-largest_bound[1])
motion=cv.CreateImage((bounding_box[2],bounding_box[3]),img_depth,1)

for fname in dirList:
    if not fnmatch.fnmatch(fname,'*.jpg'):
        continue
    fname=path+fname
    img=cv.LoadImage(fname)
    cv.SetImageROI(img,bounding_box)
    mhi_update.update_mhi(img, motion, 35)   

hu_moments=getHuMoments(cv.GetMat(mhi_update.mhi),0)
hu_array=list(hu_moments)
hu_array.append(type)
av_utils.write_array_to_file(hu_array)
cv.ShowImage('image',mhi_update.mhi)
cv.WaitKey(-1)  

