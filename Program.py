import cv
import os
import fnmatch
import time
import av_utils
import mhi_update
import classify
        
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
    

Thresh = 3

bg = './data/background2.jpg'

if os.path.exists('./output.txt'):
    os.remove('./output.txt')
    
pathes = [('./data/train/1-1/',1),('./data/train/1-2/',1),('./data/train/1-3/',1)
        ,('./data/train/1-4/',2),('./data/train/1-5/',2),('./data/train/1-6/',2)
        ,('./data/train/1-7/',3),('./data/train/1-8/',3),('./data/train/1-9/',3)
        ,('./data/train/2-1/',3),('./data/train/2-2/',1),('./data/train/2-3/',2)
        ,('./data/train/3-9/',1),('./data/train/2-5/',1),('./data/train/2-6/',2)
        ,('./data/train/3-1/',2),('./data/train/3-2/',2),('./data/train/3-3/',2)
        ,('./data/train/3-4/',3),('./data/train/3-5/',3),('./data/train/3-6/',3)
        ,('./data/train/3-7/',1),('./data/train/3-8/',1),('./data/train/2-4/',3)]
index=0
pred=-1
for path, type in pathes:
    index=index+1
    mhi=mhi_update.mhi()
    time = 10
    dirList=os.listdir(path)
    cv.NamedWindow('image')

    av_utils.av_debug('background image loaded from: '+ bg)
    bg_img = cv.LoadImage(bg)
    img_size = cv.GetSize(bg_img)
    img_depth = bg_img.depth
    img_channel = bg_img.channels

    av_utils.av_debug('image info, size:' + str(img_size) + ', ' +
                      'depth:' + str(img_depth) + ', ' +
                      'channel:' + str(img_channel))
                  
    bg_gray_img = cv.CreateImage(img_size, img_depth, 1)
    
    cv.ShowImage('image', bg_img)
    cv.WaitKey(time)

    storage = cv.CreateMemStorage(0)
    bounding_rects = list()

    last_bw_sub = False
    last_bw_sub_set = False

    maxareaseq = []
    rectimgs = []
    indexj = 0

    for fname in dirList:
        indexj = indexj + 1
        if not fnmatch.fnmatch(fname,'*.jpg'):
            continue
        
        fname = path + fname
        img = cv.LoadImage(fname)
        img_sub = cv.CreateImage(img_size, img_depth, img_channel)
        cv.Sub(img, bg_img, img_sub)
        cv.Smooth(img_sub, img_sub, cv.CV_MEDIAN, 5)
    
        img_bw_sub = cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)
    
        cv.ShowImage('image', img_sub)
        cv.InRangeS(img_sub, (0, 0, 27), (100, 100, 140), img_bw_sub)
    
        if not last_bw_sub_set:
            last_bw_sub_set = True
            last_bw_sub = img_bw_sub
            continue
        else:
            cv.Sub(img_bw_sub, last_bw_sub, img_bw_sub)
    
        img_bw_sub_dilated=cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)
        kernel=cv.CreateStructuringElementEx(11,11,6,6,cv.CV_SHAPE_CROSS)
        cv.Dilate(img_bw_sub,img_bw_sub_dilated,kernel)

        img_bw_sub_dilated_preContours = cv.CreateImage(img_size, 
                                                        cv.IPL_DEPTH_8U, 1)
        cv.Copy(img_bw_sub_dilated, img_bw_sub_dilated_preContours)
    
        seq = cv.FindContours(img_bw_sub_dilated_preContours, storage, 
                              cv.CV_CHAIN_APPROX_SIMPLE)
    
        if (len(seq) != 0):
            cnt_biggest, area = getBiggestCountour(seq)
            if (area > 15000):
                maxareaseq.append(area)
                
                rect = cv.BoundingRect(cnt_biggest)
                bounding_rects.append(rect)
                cv.DrawContours(img_sub, cnt_biggest, 
                                cv.RGB(255,255,255), 
                                cv.RGB(255,255,255), 0)
                                
                
                cv.Rectangle(img_sub, (rect[0], rect[1]), 
                                      (rect[0] + rect[2], rect[1] + rect[3]),
                                      cv.RGB(255,255,255))    
            
                tempimg = cv.CreateImage(img_size, img_depth, img_channel)
                cv.DrawContours(tempimg, cnt_biggest, 
                                cv.RGB(255,255,255), 
                                cv.RGB(255,255,255), 0, -1)
                rectimgs.append(cv.GetSubRect(tempimg, rect))
                                               
                cv.ShowImage('image', img_sub)
                cv.ShowImage('image_temp', img_bw_sub)
    

                largest_bound = (bounding_rects[0][0], 
                                 bounding_rects[0][1],
                                 bounding_rects[0][0] + bounding_rects[0][2],
                                 bounding_rects[0][1] + bounding_rects[0][3])

                for i in bounding_rects:
                    if i[0] < largest_bound[0]:
                        largest_bound = (i[0], largest_bound[1],
                                         largest_bound[2], largest_bound[3])
                    if i[1] < largest_bound[1]:
                        largest_bound = (largest_bound[0], i[1],
                                         largest_bound[2], largest_bound[3])
                    if (i[2] + i[0]) > largest_bound[2]:
                        largest_bound = (largest_bound[0], largest_bound[1],
                                         (i[0] + i[2]), largest_bound[3])
                    if (i[3]+i[1]) > largest_bound[3]:
                        largest_bound = (largest_bound[0], largest_bound[1],
                                         largest_bound[2], (i[1] + i[3]))
                                         
                bounding_box = (largest_bound[0], largest_bound[1],
                                largest_bound[2] - largest_bound[0],
                                largest_bound[3] - largest_bound[1])
                boxsize = (bounding_box[2], bounding_box[3])
                
            if (cv.WaitKey(time) == 115):
                cv.SaveImage('./results/'+str(index)+str(indexj)+str('bw.jpg'),
                             cv.GetSubRect(tempimg, rect))
                cv.SaveImage('./results/'+str(index)+str(indexj)+str('orig.jpg'),
                             cv.GetSubRect(img, rect))
                cv.SaveImage('./results/'+str(index)+str(indexj)+str('sub.jpg'),
                             cv.GetSubRect(img_sub, rect))

            

    motion = cv.CreateImage(boxsize, img_depth, 1)
    frame = 0
    timestamp = 0
    frames = len(rectimgs)
    centre = frames / 2
    for rectimg in rectimgs:
        tempimg = cv.CreateImage(boxsize, img_depth, 3)
        cv.Resize(rectimg, tempimg)
        mhi.update_mhi(tempimg, motion, 25)  
    hu_moments=getHuMoments(cv.GetMat(mhi.mhi),0)
    hu_array=list(hu_moments)

    
    for i in range(5):
        area = 0
        cnt = 0
        for ii in range(len(maxareaseq) / 5):
            j = len(maxareaseq) / 5 * i + ii
            if j < len(maxareaseq):
                cnt = cnt + 1
                area = area + maxareaseq[j]
    
        area = area / cnt
        hu_array.append(area)
    if(type==-1):
        if(pred==-1):
            model,pca_model,mean,stan= classify.script()
            classify.pred_new_instance(model,pca_model,mean,stan,hu_array)
            pred=1
        else:
            classify.pred_new_instance(model,pca_model,mean,stan,hu_array)
        
    else:
        hu_array.append(type)
        av_utils.write_array_to_file(hu_array)
        cv.ShowImage('image_mhi',mhi.mhi)
        if(cv.WaitKey(time) ==115):
            out=cv.CreateImage(boxsize, cv. IPL_DEPTH_32F, 3)
            cv.CvtColor(mhi.mhi,out,cv.CV_GRAY2BGR)
            cv.CvtScale(out,out,255)
            cv.SaveImage('./results/'+str(index)+str('mhi.jpg'),out)
if(pred==-1):
    classify.script()


