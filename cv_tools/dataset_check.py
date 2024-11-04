# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_dataset_check.ipynb.

# %% auto 0
__all__ = ['data_path', 'move_path', 'display_image_row', 'interactive_roi_selector', 'apply_roi_mask']

# %% ../nbs/03_dataset_check.ipynb 3
import os
import cv2
import shutil
from IPython.display import display, clear_output
import io
from PIL import Image as Image
from ipywidgets import (widgets, Button,
                        HBox, VBox, Layout)
from ipywidgets import Image as WImage
from pathlib import Path
import numpy as np
from fastcore.all import *
from typing import List, Union, Any, Callable

# %% ../nbs/03_dataset_check.ipynb 4
from ipywidgets.widgets import Button

# %% ../nbs/03_dataset_check.ipynb 5
def display_image_row(
        im_path:Union[Path, str],
        move_path:Union[Path, str],
        max_images:int=10,
        start:int=0,
        im_height:int=100,
        im_width:int=100,
    ): 


    Path(move_path).mkdir(exist_ok=True, parents=True)

    def on_next_click(b):
        nonlocal start

        start = start + max_images
        print(f' start = {start}')
        clear_output(wait=True)
        display_image_row(
            im_path,
            move_path,
            max_images,
            start, 
            im_height,
            im_width
        )
    def on_prev_click(b):
        nonlocal start
        start = max( 0, start - max_images)
        clear_output(wait=True)
        display_image_row(
            im_path,
            move_path,
            max_images,
            start,
            im_height,
            im_width
        )
    def on_move_click(
            b,
            file_name,
        ):
        source_path = Path(im_path, file_name)
        destination_path = Path(move_path, file_name)
        shutil.move(source_path, destination_path)
        clear_output(wait=True)
        display_image_row(
            im_path,
            move_path,
            max_images,
            start,
            im_height,
            im_width
        )
    clear_output(wait=True)

    files = sorted(im_path.ls(file_exts=['.png', '.tif']))
    files_to_d = files[start:start + max_images]

    image_boxes = []
    
    for file in files_to_d:
        name_ = Path(file).name
        image_path = Path(im_path, name_)

        with Image.open(image_path) as pil_img:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_widget = WImage(
                value=img_byte_arr.getvalue(),
                format='png',
                width=im_width,
                height=im_height
                )
            
            btn_del = widgets.Button(description='Delete')
            btn_move = widgets.Button(description='Move')

            btn_move.on_click(lambda b,f=name_: on_move_click(b, f))

            box = VBox(
                [
                    img_widget,
                    HBox([btn_del, btn_move])
                    ])
            image_boxes.append(box)


    btn_next = Button(description='Next') 
    btn_prev = Button(description='Prev') 

    if start > 0 : 
        print(f'start = {start}')
        print(f'max = {max_images}')
        btn_prev.on_click(on_prev_click)
    if start + max_images < len(files):
        btn_next.on_click(on_next_click)

    nav = HBox([btn_prev, btn_next])
    display(
        HBox(image_boxes, Layout=Layout(
            flex_flow='row_wrap', 
            align_items='center',
        ))
    )
    display(nav)
data_path = Path(r'N:\homes\hasan\easy_front\overlay_path')
move_path = Path(r'N:\homes\hasan\easy_front\overlay_path_old')


# %% ../nbs/03_dataset_check.ipynb 11
def interactive_roi_selector(img):
    """Create an interactive ROI selector that returns multiple ROI coordinates and masked image
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        tuple: (roi_coords_list, masked_image) where roi_coords_list is a list of (x,y,w,h) coordinates
               and masked_image has everything outside all ROIs blacked out
    """
    roi_coords_list = []  # Initialize empty list to store multiple ROI coordinates
    drawing = False  # Flag to track if we're currently drawing
    ix,iy = -1,-1   # Initialize starting coordinates
    img_display = img.copy()  # Create copy for display
    
    def draw_roi(event,x,y,flags,param):
        nonlocal ix,iy,drawing,img_display,roi_coords_list
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_temp = img_display.copy()
                cv2.rectangle(img_temp,(ix,iy),(x,y),(0,255,0),2)
                cv2.imshow('image',img_temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            w = abs(x - ix)
            h = abs(y - iy)
            x_start = min(ix, x)
            y_start = min(iy, y)
            roi_coords_list.append((x_start, y_start, w, h))
            cv2.rectangle(img_display,(x_start,y_start),(x_start+w,y_start+h),(0,255,0),2)
            cv2.imshow('image',img_display)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_roi)

    print("Select ROIs by dragging rectangles. Press 'r' to reset, ESC when done.")
    while(1):
        cv2.imshow('image',img_display)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k == ord('r'):  # Reset
            img_display = img.copy()
            roi_coords_list = []
            
    cv2.destroyAllWindows()
    
    if not roi_coords_list:
        return None, None
        
    # Create masked image combining all ROIs
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x,y,w,h in roi_coords_list:
        mask[y:y+h, x:x+w] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    return roi_coords_list, masked_img

def apply_roi_mask(img, roi_coords_list):
    """Apply multiple ROI masks to an image using saved coordinates
    
    Args:
        img: Input image
        roi_coords_list: List of (x,y,w,h) coordinates for each ROI
        
    Returns:
        numpy array: Masked image with everything outside ROIs blacked out
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x,y,w,h in roi_coords_list:
        mask[y:y+h, x:x+w] = 255
    return cv2.bitwise_and(img, img, mask=mask)

