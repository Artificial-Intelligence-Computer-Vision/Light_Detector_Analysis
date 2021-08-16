import cv2
import numpy as np
from glob import glob
import os
from os.path import basename




class light_detection_analysis(object):
    def __init__(self):
        self.image_path = "images_data/"
        self.images = [count for count in glob(image_path +'*') if 'jpg' in count]

        # Count from the standard
        self.count = 0


    def save_image(img, file_name, image_to_save):
        image_path = "images_data/"
        image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
        if image_to_save == 'brightness_detector':
            image_output = "brightness_detector/"
        
        for i in range(len(image_number)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), img)
            cv2.waitKey(0)
        
    
    def image_looping(self):
        
        for image in images:
            self.count +=1
            print(self.count)
            img = cv2.imread(image, -1)
            file_name = basename(image)
    
            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_scale, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)

            brightness_detector = highlights_brightness(img)
            brightness_detector[mask == 0] = [255, 0, 0]
            save_image(brightness_detector, file_name, image_to_save = "brightness_detector")



    
    def highlights_brightness(img):

        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        img[:,:,1] = img[:,:,1] + (0.01 + np.random.normal())
        img_modify = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    
        return img_modify

    

    def light_source_detection(img):
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fill any small holes
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations = 2)

        # threshold (remove grey shadows)
        dilation[dilation < 240] = 0
        #=========================== contours ======================
        im, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # extract every contour and its information:
        for cID, contour in enumerate(contours):
            M = cv2.moments(contour)
            # neglect small contours:
            if M['m00'] < 400:
                continue
            # centroid
            c_centroid = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

            # area
            c_area = M['m00']
            # perimeter
            try:
                c_perimeter = cv2.arcLength(contour, True)
            except:
                c_perimeter = cv2.arcLength(contour, False)
            # convexity
            c_convexity = cv2.isContourConvex(contour)
            # boundingRect
            (x, y, w, h) = cv2.boundingRect(contour)
            # br centroid
            br_centroid = (x + int(w/2), y + int(h/2)) 
            # draw rect for each contour: 
            cv2.rectangle(original_frame,(x,y),(x+w,y+h),(0,255,0),2)
            # draw id:
            cv2.putText(original_frame, str(cID), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, (127, 255, 255), 1)
            # save contour info
            contours_info.append([cID,frameID,c_centroid,br_centroid,c_area,c_perimeter,c_convexity,w,h])


        #======================= show processed frame img ============================
        cv2.imshow('fg',dilation)
        cv2.imshow('origin',original_frame)
        # save frame image:
        cv2.imwrite('pics/{}.png'.format(str(frameID)), original_frame)
        cv2.imwrite('pics/fb-{}.png'.format(str(frameID)), dilation)
        frameID += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            break
    
    
