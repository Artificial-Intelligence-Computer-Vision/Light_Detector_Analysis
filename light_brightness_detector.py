from header_imports import *


class light_detection_analysis(object):
    def __init__(self):
        self.image_path = "images_data/"
        self.images = [count for count in glob(self.image_path +'*') if 'jpg' in count]

        # Kernel Creation 
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Count from the standard
        self.count = 0
        
        # Start analysing
        self.image_looping()
        
        # Mask
        self.mask = None
        
        # Image
        self.contours_info = [];



    def save_image(self, img, file_name, image_to_save):
        image_path = "images_data/"
        image_number = [count for count in glob(self.image_path + '*') if 'jpg' in count]
        
        if image_to_save == 'brightness_detector':
            image_output = "brightness_detector/"
        elif image_to_save == 'light_detection_source':
            image_output = "light_detection_source"
        elif image_to_save == "light_detection":
            image_output = "light_detection"


        
        for i in range(len(image_number)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), img)
            cv2.waitKey(0)
        

    
    def image_looping(self):
        
        for image in self.images:
            self.count +=1
            print(self.count)

            img = cv2.imread(image, -1)
            file_name = basename(image)
    
            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_scale, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            self.mask = mask

            brightness_detector = self.highlights_brightness(img)
            light_source_detector = self.light_source_detection(img)
            light_detection = self.light_detection(brightness_detector)

            self.save_image(brightness_detector, file_name, image_to_save = "brightness_detector")
            self.save_image(light_source_detector, file_name, image_to_save = "light_detection_source")
            self.save_image(light_detection, file_name, image_to_save = "light_detection")



    
    def highlights_brightness(self, img):

        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        img[:,:,1] = img[:,:,1] + (0.01 + np.random.normal())
        img_modify = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

        # Change the color wanted to display
        img_modify[self.mask == 0] = [255, 0, 0]

        return img_modify

    

    def light_source_detection(self, img):

        original_frame = img.copy()
        shadow = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
        mask = shadow.apply(img)
    
        # Fill any small holes
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, self.kernel)

        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, self.kernel, iterations = 2)

        # Threshold (remove grey shadows)
        dilation[dilation < 150] = 0

        # Contours
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract every contour and the information:
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

            # Convexity
            c_convexity = cv2.isContourConvex(contour)

            # BoundingRect
            (x, y, w, h) = cv2.boundingRect(contour)

            # Br centroid
            br_centroid = (x + int(w/2), y + int(h/2))

            # Draw rect for each contour: 
            cv2.rectangle(original_frame,(x,y),(x+w,y+h),(0,255,0),2)

            # Draw id:
            cv2.putText(original_frame, str(cID), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, (127, 255, 255), 1)

            # Save contour info
            # self.contours_info.append([cID,frameID,c_centroid,br_centroid,c_area,c_perimeter,c_convexity,w,h])

        return original_frame




    def light_detection(self, img):

        col_switch = cv2.cvtColor(img, 70)
        
        # Upperbound and lower bounds
        lower = np.array([0,0,0])
        upper = np.array([40,10,255]) 

        shadow = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
        mask = cv2.inRange(col_switch, lower, upper)
        res = cv2.bitwise_and(col_switch,col_switch, mask= mask)

        fgmask = shadow.apply(res)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Dilate to merge adjacent blobs
        d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(fgmask, d_kernel, iterations = 2)
        dilation[dilation < 255] = 0

        return dilation
