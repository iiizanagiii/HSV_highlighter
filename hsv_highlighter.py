import os
import pickle
import time
import PIL.Image, PIL.ImageTk
import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk, Image
import yaml
import glob
from pathlib import Path
from tkinter.filedialog import askdirectory


FILE = Path(__file__).parent
INPUT = Path(FILE / 'input')

# PICKLE_DIR = os.path.join()


class TestColorHighlight:
    def __init__(self) -> None:
        self.hue_min_val = 0
        self.hue_max_val = 0 
    
    def load_image(self, image_path, mask_path):
        image = cv2.imread(image_path)
        image_mask = cv2.imread(mask_path)
        threshold = 100
        image_mask[image_mask > threshold]  = 255
        image_mask[image_mask <= threshold] = 0
        image = cv2.bitwise_and(image, image_mask)
        return image 

    def calibrate_hue_value(self):
        min_hue_values = []
        max_hue_values = [] 
        import matplotlib.pyplot as plt 
        from scipy import stats
        import pandas as pd

        h_mean = [] 
        s_mean = [] 
        v_mean = [] 

        h_std = []
        s_std = []
        v_std = [] 

        for img_path, msk_path in zip(sorted(glob.glob(self.path_to_segment_image)), sorted(glob.glob(self.path_to_segment_mask))):
            print('[+] ', img_path, msk_path)
            img = self.load_image(img_path, msk_path)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            h = h.ravel() 
            s = s.ravel()
            v = v.ravel() 

            if np.any(h>1):
                s = s[h>0]
                v = v[h>0]
                h = h[h>0]
                
                df = pd.DataFrame({'h': h, 's': s, 'v': v })
                print('[X-X] ', df.describe())

                h_mean.append(df.describe()['h']['mean'])
                s_mean.append(df.describe()['s']['mean'])
                v_mean.append(df.describe()['v']['mean'])

                h_std.append(df.describe()['h']['std'])
                s_std.append(df.describe()['s']['std'])
                v_std.append(df.describe()['v']['std'])
                
                # cv2.imshow('image ', img)
                # cv2.imshow('hsv', hsv_img)
                
                print('[X] h value ', h.shape, h)
                h = h[h> 1]
                print('[X] h value ', h.shape)
                
                v = v[v > 25] 
                plt.hist(v)
                plt.savefig('test.png')
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                low_percentile, high_percentile = 25, 95

                hue_low = np.percentile(h, low_percentile)
                hue_high = np.percentile(h, high_percentile)
                print('[+] H ', type(img), h.shape, np.average(h), hue_low, hue_high)
                print('[+] S ', type(img), s.shape, np.average(s), hue_low, hue_high)
                print('[+] V ', type(img), v.shape, np.average(v), hue_low, hue_high)
                
                min_hue_values.append(hue_low)
                max_hue_values.append(hue_high)




        # self.hue_min_val = np.average(min_hue_values) # min(min_hue_values)
        # self.hue_max_val = np.average(max_hue_values)
        
        average_h_mean = np.average(h_mean)
        average_s_mean = np.average(s_mean)
        average_v_mean = np.average(v_mean)

        average_h_std = max(h_std)
        average_s_std = max(s_std)
        average_v_std = max(v_std)

        sigma_h = 0.5
        sigma_s = 0.5
        sigma_v = 0.5

        self.hue_min_val = 0 if (average_h_mean - sigma_h * average_h_std) < 0 else (average_h_mean - sigma_h * average_h_std)
        self.hue_max_val = 179 if (average_h_mean + sigma_h * average_h_std) > 179 else (average_h_mean + sigma_h * average_h_std)
        
        self.saturate_min_val = 0 if (average_s_mean -  sigma_s * average_s_std)< 0 else (average_s_mean - sigma_s * average_s_std)
        self.saturate_max_val = 255 if (average_s_mean + sigma_s * average_s_std)>255 else (average_s_mean + sigma_s * average_s_std)
        
        self.value_min_val = 0 if (average_v_mean - sigma_v * average_v_std) < 0 else (average_v_mean - sigma_v * average_v_std)
        self.value_max_val = 255 if (average_v_mean + sigma_v * average_v_std) > 255 else (average_v_mean + sigma_v * average_v_std)
        
        print('values ', min_hue_values, max_hue_values, self.hue_min_val, self.hue_max_val)

    def find_hue_color(self, image: np.array):
        current_hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[(current_hsv_image[:, :, 0] > (self.hue_min_val)) & (current_hsv_image[:, :, 0] < (self.hue_max_val)) 
            # & (current_hsv_image[:, :, 1] > self.saturate_min_val) & (current_hsv_image[:, :, 1] < self.saturate_max_val) 
            # & (current_hsv_image[:, :, 2] > self.value_min_val) & (current_hsv_image[:, :, 2] < self.value_max_val) 
            ] = [0, 255, 255]
        
        # image[np.logical_not((current_hsv_image[:, :, 0] > self.hue_min_val) & (current_hsv_image[:, :, 0] < self.hue_max_val))] = [0, 0, 0]

        return image 
    
    
class Imager:
    # Define the necessary variables
    tolerance = 1
    listG = []
    fg_range = [] 
    all_images_list = []    
    drawing = False
    erase = False
    color_green = (0, 0, 255) 
    alpha = 0.9
    beta = 0.9
    gamma = 0.9
    multiple_masks = []
    current_index = 0
    current_original_image = 0 
    current_mask_image = 0
    scale_percent = 50
    directory = FILE
    pickle_path = os.path.join(FILE, 'hsv_highlighter.pkl')
    diff_thresh = 30
    
    # Save the initial empty list to the pickle file
    with open(pickle_path, 'wb') as f:
        pickle.dump(multiple_masks, f)
    
    def __init__(self, directoy):
        self.load_all_images(directoy) 
        self.load_image()
    
    def load_all_images(self,dir_path ):
        for image_path in os.listdir(dir_path):
            if '.jpg' in image_path:
                self.all_images_list.append(os.path.join(dir_path, str(image_path)))
    
    def load_image(self):        
        if self.current_original_image is not None:
            self.current_original_image = cv2.imread(self.all_images_list[self.current_index])
            self.current_mask_image = np.zeros_like(self.current_original_image[:, :, 0])
            width = int(self.current_original_image.shape[1] * self.scale_percent / 100)
            height = int(self.current_original_image.shape[0] * self.scale_percent / 100)
            dim = (width, height)
            self.current_original_image = cv2.resize(self.current_original_image,dim, cv2.INTER_AREA)
            self.img_copied = self.current_original_image.copy()
        else:
            pass
        
    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.all_images_list)
        self.load_image()
        self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
        self.current_original_image = cv2.addWeighted(self.current_original_image, self.alpha, self.mask_img_fg, self.beta, self.gamma)
        self.refresh()

    def previous_image(self):
        self.current_index = (self.current_index - 1 + len(self.all_images_list)) % len(self.all_images_list)
        self.load_image()
        self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
        self.current_original_image = cv2.addWeighted(self.current_original_image, self.alpha, self.mask_img_fg, self.beta, self.gamma)
        self.refresh() 
    
    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            list_hsv = self.hsv_finder(self.img_copied, y, x) # get [h, s, v] in list_hsv
            self.hsv_(list_hsv) # remove duplicate from list_hsv
            masks = self.hsv_ranging(self.listG) # use tolerance for combination of hsv values. 
                                                 # eg: [[7,7,7], [9,9,9]] -> here tolerence=1 for [8,8,8]

            self.multiple_masks = self.hsv_comparing_with_new_mask(masks, self.multiple_masks) # compare new mask with the list of masks in self.multiple_masks. 
                                                                                               # ie-> new_mask = [[12,46,98],[15,49,102]]
            
            self.multiple_masks = self.hsv_comparing_within_the_list(self.multiple_masks) # compare masks within self.multiple_masks. 
                                                                                          # ie-> self.multiple_masks = [[[12,46,98],[15,49,102]], [[1,2,3],[3,4,5]], ...]
            
            self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green) # masking the ranges from self.multiple using cv2.inRange
            
            self.current_original_image = cv2.addWeighted(self.img_copied, self.alpha, self.mask_img_fg, self.beta, self.gamma) # addweight for blending the mask in original image
            
            self.append_to_pickle_file(self.multiple_masks) # append the self.multiple_masks in pickle

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erase = True
            self.ix, self.iy = x, y 
            self.multiple_masks.pop()
            self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
            self.current_original_image = cv2.addWeighted(self.img_copied, self.alpha, self.mask_img_fg, self.beta, self.gamma) 
        
        elif event == cv2.EVENT_RBUTTONUP:
            self.erase = False
    
    def change_tolerance(self, value): # tolerance in trackbar
        self.tolerance = value if value != 0 else 1

    def hsv_ranging(self, hsv_values): # use tolerance for combination of hsv values. 
                                       # eg: [[7,7,7], [9,9,9]] -> here tolerence=1 for [8,8,8]
        last_hsv_value = hsv_values[-1]
        all_ranges = []
        lower_range = (last_hsv_value[0] - self.tolerance, last_hsv_value[1] - self.tolerance, last_hsv_value[2] - self.tolerance)
        upper_range = (last_hsv_value[0] + self.tolerance, last_hsv_value[1] + self.tolerance, last_hsv_value[2] + self.tolerance)
        ranges = []
        for i in range(lower_range[0], upper_range[0] + 1):
            for j in range(lower_range[1], upper_range[1] + 1):
                for k in range(lower_range[2], upper_range[2] + 1):
                    ranges.append((i, j, k))
        all_ranges.extend(ranges)
        unique_ranges = np.unique(np.array([sub_list for sub_list in all_ranges]),axis=0)
        unique_ranges = unique_ranges.tolist()
        all_ranges = [t for t in all_ranges if all(i >= 0 for i in t) if t[0] <= 179 if t[1] <= 255 if t[2] <= 255]
        unique_ranges = [t for t in unique_ranges if all(i >= 0 for i in t) if t[0] <= 179 if t[1] <= 255 if t[2] <= 255]
        masks = [min(unique_ranges), max(unique_ranges)]

        return masks
    
    def hsv_comparing_with_new_mask(self, masks, mask_list): # compare new mask with the list of masks in self.multiple_masks. 
                                                                # ie-> new_mask = [[12,46,98],[15,49,102]]
        append_new_masks = 0
        low_h1, low_s1, low_v1 = masks[0][0], masks[0][1], masks[0][2]
        high_h1, high_s1, high_v1 = masks[1][0], masks[1][1], masks[1][2]        
        if len(mask_list) > 0:
            for item in mask_list:
                low_h, low_s, low_v = item[0][0], item[0][1], item[0][2]
                high_h, high_s, high_v = item[1][0], item[1][1], item[1][2]
                
                compare_low = lambda low_x, low_x1 : low_x1 if (low_x - low_x1) < self.diff_thresh and (low_x - low_x1) > 0 else low_x
                compare_high = lambda high_x, high_x1 : high_x if (high_x-high_x1) < self.diff_thresh and (high_x - high_x1) > 0 else high_x1
                
                if (abs(low_h - low_h1) < self.diff_thresh and abs(high_h - high_h1) < self.diff_thresh 
                    and abs(low_s - low_s1) < self.diff_thresh and abs(high_s - high_s1) < self.diff_thresh 
                    and abs(low_v - low_v1) < self.diff_thresh and abs(high_v - high_v1) < self.diff_thresh):
                    low_h = compare_low(low_h, low_h1)
                    low_s = compare_low(low_s, low_s1)
                    low_v = compare_low(low_v, low_v1)
                    
                    high_h = compare_high(high_h, high_h1)
                    high_s = compare_high(high_s, high_s1)
                    high_v = compare_high(high_v, high_v1)

                    item[0][0], item[0][1], item[0][2] = low_h, low_s, low_v
                    item[1][0], item[1][1], item[1][2] = high_h, high_s, high_v 
                    append_new_masks += 1

                else:
                    None
        
        if append_new_masks == 0:
            mask_list.append(masks)
            
        return mask_list 
    
    def hsv_comparing_within_the_list(self, mask_list): # compare masks within self.multiple_masks. 
                                                            # ie-> self.multiple_masks = [[[12,46,98],[15,49,102]], [[1,2,3],[3,4,5]], ...]
        from copy import deepcopy
        
        low, high = 0 , 1
        h, s, v = 0 , 1 , 2
        index_to_remove = []
        
        temp_lis = deepcopy(mask_list)

        for i_index in range(0,len(mask_list)):
            i = mask_list[i_index]
            low_h1, low_s1, low_v1 = i[low][h], i[low][s], i[low][v]
            high_h1, high_s1, high_v1 = i[high][h], i[high][s], i[high][v]

            
            for j_index in range(i_index + 1,len(mask_list)):
                j = mask_list[j_index]
            
                low_h2, low_s2, low_v2 = j[low][h], j[low][s], j[low][v]
                high_h2, high_s2, high_v2 = j[high][h], j[high][s], j[high][v]
                
                if (abs(low_h1 - low_h2) < self.diff_thresh and abs(high_h1 - high_h2) < self.diff_thresh 
                    and abs(low_s1 - low_s2) < self.diff_thresh and abs(high_s1 - high_s2) < self.diff_thresh 
                    and abs(low_v1 - low_v2) < self.diff_thresh and abs(high_v1 - high_v2) < self.diff_thresh ):

                    tp = temp_lis[i_index]
                
                    tp[low][h] = min(low_h1 , low_h2)
                    tp[low][s] = min(low_s1 , low_s2)
                    tp[low][v] = min(low_v1 , low_v2)
                    
                    tp[high][h] = max(high_h1 , high_h2)
                    tp[high][s] = max(high_s1 , high_s2)
                    tp[high][v] = max(high_v1 , high_v2)

                    index_to_remove.append(j)

        for ix in index_to_remove[::-1]:
            try:
                temp_lis.remove(ix)
            except ValueError as e:
                print(f'value {ix} not in list{temp_lis}')
                
        if len(index_to_remove):
            temp_lis = self.hsv_comparing_within_the_list(temp_lis)
        
        return temp_lis

    def highlighter(self, mask_list, color): # masking the ranges from self.multiple using cv2.inRange
        
        # masks = [list(tuple(j) for j in i) for i in set(tuple(tuple(i) for i in j) for j in mask_list)]
        masks = [inner_list for i, inner_list in enumerate(mask_list) if inner_list not in mask_list[:i]]
        combined_mask = np.zeros_like(self.img_copied[:, :, 0])
        for mask in masks:
            low = np.array(mask[0])
            up = np.array(mask[1])
            hsv_image = cv2.cvtColor(self.img_copied, cv2.COLOR_BGR2HSV)
            temp_mask = cv2.inRange(hsv_image, low, up)
            combined_mask = cv2.bitwise_or(combined_mask, temp_mask)
        green_image = np.zeros_like(self.img_copied)
        green_image[:] = color
        masked_image = cv2.bitwise_and(self.img_copied, self.img_copied, mask=combined_mask)
        green_image = cv2.bitwise_or(green_image, green_image, mask=combined_mask)
        
        return masked_image, green_image

    def load_from_pickle_file(self): # load self.multiple_masks from pickle
        with open(self.pickle_path, 'rb') as f1:
            obj1 = pickle.load(f1)
        
        return obj1

    def append_to_pickle_file(self, mask_list): # append self.multiple_masks in pickle
        
        # masks = [list(tuple(j) for j in i) for i in set(tuple(tuple(i) for i in j) for j in mask_list)]
        masks = [inner_list for i, inner_list in enumerate(mask_list) if inner_list not in mask_list[:i]]
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(masks, f)

    def hsv_(self, list_hsv):
        if len(self.listG) < 0:
            pass
        else:
            self.listG.append(list_hsv)
            self.listG = [inner_list for i, inner_list in enumerate (self.listG) if inner_list not in self.listG[:i]] 

    @staticmethod
    def hsv_finder(output, y, x):
        colors = output[y, x]
        colors = np.array([[[colors[0], colors[1], colors[2]]]])
        hsv = cv2.cvtColor(colors, cv2.COLOR_BGR2HSV)
        hsvlist = hsv.tolist()
        h = hsvlist[0][0][0]
        s = hsvlist[0][0][1]
        v = hsvlist[0][0][2]
        return h, s, v
    
    def label_images(self):
        original_canvas = cv2.resize(self.img_copied,(255,255))
        original_canvas = cv2.cvtColor(original_canvas , cv2.COLOR_BGR2RGB)
        masked_resized_2_canvas = cv2.resize(self.current_original_image, (255, 255))
        masked_resized_2_canvas = cv2.cvtColor(masked_resized_2_canvas, cv2.COLOR_BGR2RGB)
        original_photoimage = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(original_canvas))
        masked_photoimage = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(masked_resized_2_canvas))

        return original_photoimage,masked_photoimage
    
    def set_label_images(self,canvas,original_image_canvas,masked_image_canvas):
        self.canvas = canvas
        self.original_image_canvas = original_image_canvas
        self.masked_image_canvas = masked_image_canvas 
        
    def refresh(self):
        original_photoimage,masked_photoimage = self.label_images()
        self.canvas.itemconfig(self.original_image_canvas,image = original_photoimage)
        self.canvas.itemconfig(self.masked_image_canvas, image = masked_photoimage)
        self.canvas.image1 = original_photoimage
        self.canvas.image2 = masked_photoimage 
        
    def run(self):
        self.load_image()
        self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
        self.current_original_image = cv2.addWeighted(self.current_original_image, self.alpha, self.mask_img_fg, self.beta, self.gamma)
        cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback('image', self.draw_circle)
        cv2.createTrackbar('Tolerance', 'image', self.tolerance, 5, self.change_tolerance)
        while True:
            masked_and_original_image = cv2.hconcat([self.current_original_image, self.current_mask_image])
            if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:
                break
            else:
                cv2.imshow('image', masked_and_original_image)
                
            k = cv2.waitKey(1)
            if k == 27:
                break 
        cv2.destroyAllWindows()
        self.refresh()
        
if __name__ == '__main__':
    imager = Imager(INPUT)

    some_img = cv2.resize(imager.current_mask_image, (255,255))
    h, w = some_img.shape[:2]

    root = tk.Tk()
    canvas = tk.Canvas(root,width=520, height=270)
    canvas.pack()
    icon = Image.open('./assets/highlighter_icon.png')
    photo = ImageTk.PhotoImage(icon)

    next_button = tk.Button(root, text=">>", command=imager.next_image, bg='red', fg='white')
    edit_button = tk.Button(root, image = photo, command=imager.run)
    prev_btn = tk.Button(root, text="<<", command=imager.previous_image, bg='red', fg='white')

    # refresh_btn.pack(padx=5, pady=15, side=tk.LEFT)
    next_button.pack(padx=5, pady=15, side=tk.RIGHT)
    edit_button.pack(padx=5, pady=15, side=tk.RIGHT)
    prev_btn.pack(padx=5, pady=15, side=tk.RIGHT)

    original_photoimage,masked_photoimage = imager.label_images()
    original_image_canvas = canvas.create_image(0, 0, image=original_photoimage, anchor=tk.NW)
    masked_image_canvas = canvas.create_image(w+10, 0, image=masked_photoimage, anchor=tk.NW)
    imager.set_label_images(canvas, original_image_canvas,masked_image_canvas)

    root.mainloop()