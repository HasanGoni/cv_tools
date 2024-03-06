# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['get_name_', 'dpi', 'label_mask', 'show_labeled_mask', 'write_new_mask', 'remove_object_from_mask', 'read_img',
           'show_', 'overlay_mask', 'overlay_mask_border_on_image', 'concat_images', 'show_poster_from_path',
           'get_template_part', 'split_image', 'find_contours_binary', 'foo']

# %% ../nbs/00_core.ipynb 2
from PIL import Image
import cv2
from fastcore.all import *
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from tqdm.auto import tqdm
from typing import Union, Dict, List
#import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import (
    label, binary_dilation, binary_erosion,label,
    )
from skimage.color import label2rgb
import matplotlib as mpl

# %% ../nbs/00_core.ipynb 3
get_name_ = np.vectorize(lambda x: x.name)
dpi = mpl.rcParams['figure.dpi']

# %% ../nbs/00_core.ipynb 4
def label_mask(
        mask:np.ndarray
        )->np.ndarray:
    "Label connected components in a binary mask"
    try: 
        labels, num_labels = label(mask)
        return labels, num_labels
    except Exception as e:
        print(e)


# %% ../nbs/00_core.ipynb 5
def show_labeled_mask(
        msk_path):
    
    msk_img = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

    labels, num_labels = label_mask(msk_img)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(labels)
    # Annotate each object with its label number
    ax.set_facecolor('black')
    for i in range(1, num_labels + 1):
        y, x = np.where(labels == i)
        centroid = (x.mean(), y.mean())
        ax.text(centroid[0], centroid[1], str(i), color='white', ha='center', va='center', fontsize=12)
    plt.axis('off')
    return labels, num_labels

# %% ../nbs/00_core.ipynb 6
def write_new_mask(
                     new_lb, # labeled masks
                     new_mask_path, # path to save the new mask
                     fn # name of the image
                     ):
    'Write mask to new_mask_path'
    new_lbl = new_lb.astype(np.uint8)
    new_lbl = np.where(new_lbl > .9, 255, 0)
    cv2.imwrite(f'{new_mask_path}/{fn}', new_lbl)

# %% ../nbs/00_core.ipynb 7
def remove_object_from_mask(
                           mask:np.ndarray,
                           object_id_list:List[int]
                           ):
    """Remove object from mask."""

    for i in object_id_list:
        mask[mask == i] = 0
    return mask

# %% ../nbs/00_core.ipynb 9
def read_img(
    im_path:Union[str, Path],
    cv:bool=True,
    gray:bool=True
    ):
    'Read image from name could be open cv or pil image'
    if cv:
        if gray:
            return cv2.imread(f'{im_path}', cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.cvtColor(cv2.imread(f'{im_path}'), cv2.COLOR_BGR2RGB)
    
    return Image.open(im_path)

# %% ../nbs/00_core.ipynb 10
def show_(
    im_path:Union[str, np.ndarray]
    ):
    'Showing an image could be image or str, '

    if isinstance(im_path,str) or isinstance(im_path, Path):
        im = Image.open(im_path)
        fig, ax = plt.subplots(figsize= (im.size[0]/dpi, im.size[1]/dpi))
        ax.imshow(im)
        ax.axis('off')
    elif isinstance(im_path, list):
            for i in im_path:
                if isinstance(i, str) or isinstance(i, Path):
                    im = Image.open(i)
                    h, w = im.size
                    fig, ax = plt.subplots(figsize= (w/dpi, h/dpi))
                    ax.imshow(i)
                    ax.axis('off')
                elif isinstance(i, np.ndarray):
                    h, w = i.shape[0], i.shape[1]
                    fig, ax = plt.subplots(figsize= (w/dpi, h/dpi))
                    ax.imshow(i)
                    ax.axis('off')
                    
                    
    else: 
        h, w = im_path.shape[0], im_path.shape[1]
        fig, ax = plt.subplots(figsize= (w/dpi, h/dpi))
        im = im_path
        ax.imshow(im)
        ax.axis('off')

# %% ../nbs/00_core.ipynb 11
#def normalize(
              #image:Union[np.ndarray, tf.Tensor], 
              #min=0):
    #def _normalize(im):
        #img = tf.cast(im, tf.float32)
        #return img / 255.0

    #if min == 0:
        #return _normalize(image)
    #else:
        #return (_normalize(image) * 2.0) -1.0

# %% ../nbs/00_core.ipynb 13
def overlay_mask(
        im_path:Union[str,Path],
        msk_path:Union[str,Path], 
        overlay_clr:Tuple[int, int, int]=(0, 1, 0), # color
        scale:int=1, # to scale the image 
        alpha:float=0.5, # visibility
        ):
    'Creaete a overlay image from image and mask'
    # Read the grayscale image
    gray_img = cv2.imread(f'{im_path}', cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise ValueError("Could not read the grayscale image")

    # Read the mask image
    mask_img = cv2.imread(f'{msk_path}', cv2.IMREAD_GRAYSCALE)
    mask_img = mask_img.astype(bool)
    if mask_img is None:
        raise ValueError("Could not read the mask image")

    # Check if dimensions of both images are the same
    if gray_img.shape != mask_img.shape:
        raise ValueError("Dimensions of grayscale image and mask do not match")

    # Convert image to 3 channels
    rgb_img = np.stack([gray_img]*3, axis=-1)/255
    fig, ax = plt.subplots()
    ax.imshow(rgb_img)

    clrd_overlay = np.zeros_like(rgb_img)
    clrd_overlay[mask_img]=overlay_clr
    ax.imshow(clrd_overlay, alpha=alpha)

# %% ../nbs/00_core.ipynb 14
def overlay_mask_border_on_image(
        im_path: Union[Path, str],
        msk_path: Union[Path, str],
        save_new_img_path:Union[Path,str]=None,
        save_overlay_img_path:Union[Path,str]=None,
        new_img:Union[List, np.ndarray, None] = None,
        scale_:int=1,
        border_color: Tuple[int, int, int] = (0, 1, 0),
        border_width: int = 1,
        show_:bool=False):
    """
    Overlays the border of a binary mask on a grayscale image and displays the result using matplotlib.

    Args:
    image (numpy.ndarray): Grayscale image.
    mask (numpy.ndarray): Binary mask of the same size as the image.
    border_color (tuple): RGB color for the mask border in the range [0, 1].
    border_width (int): Width of the border.

    Returns:
    None: The function displays a plot.
    """
    name_ = Path(im_path).name
    gray_img = cv2.imread(f'{im_path}', cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise ValueError("Could not read the grayscale image")

    # Read the mask image
    mask_img = cv2.imread(f'{msk_path}', cv2.IMREAD_GRAYSCALE)
    mask_img = mask_img.astype(bool)
    if mask_img is None:
        raise ValueError("Could not read the mask image")

    # Check if dimensions of both images are the same
    if gray_img.shape != mask_img.shape:
        raise ValueError("Dimensions of grayscale image and mask do not match")
    # Ensure the mask is boolean

    # Find the borders of the mask
    dilated_mask = binary_dilation(mask_img, iterations=border_width)
    eroded_mask = binary_erosion(mask_img, iterations=border_width)
    border = dilated_mask & ~eroded_mask

    # Convert grayscale image to RGB
    rgb_image = np.stack([gray_img]*3, axis=-1) / \
        255.0  # Normalize for matplotlib

    # Apply the colored border
    rgb_image[border] = border_color
    rgb_image = (rgb_image * 255).astype(np.uint8)

    if new_img is not None:
        new_img = np.concatenate([rgb_image, new_img], axis=1)
        new_img = new_img.astype(np.uint8)
        if save_new_img_path is not None:
            cv2.imwrite(f'{save_new_img_path}/{name_}', new_img)
        if show_:
            fig, ax = plt.subplots(figsize=(scale_*new_img.shape[1] / dpi, scale_*new_img.shape[0] / dpi))
            ax.imshow(new_img, cmap='gray')
            ax.axis('off')  # Turn off axis numbers
    else:
        if show_:
            fig, ax = plt.subplots(figsize=(scale_*rgb_image.shape[1] / dpi, scale_*rgb_image.shape[0] / dpi))
            ax.imshow(rgb_image, cmap='gray')
            ax.axis('off')  # Turn off axis numbers
        if save_overlay_img_path is not None:
            cv2.imwrite(f'{save_overlay_img_path}/{name_}', rgb_image)



# %% ../nbs/00_core.ipynb 15
def concat_images(
        images:List[np.ndarray],
        rows:int,  # number of rows
        cols:int,  # number of columns in combined images
        number:str # a text which will be inserted in the combined image
        ):
    'Concate images rows and cols and add a number to the image.'
    targe_h = min([i.shape[0] for i in images])
    target_w = min([i.shape[1] for i in images])
    res_img = [cv2.resize(i, (target_w, targe_h)) for i in images]
    res = []
    for i in range(rows):
        start_index = i * cols
        end_index = start_index + cols
        row_images = res_img[start_index:end_index]
        row_concat = cv2.hconcat(row_images)
        res.append(row_concat)
    new_img = cv2.vconcat(res)
    position = (6, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale=0.6
    color = (255, 0, 0)
    im_n = cv2.putText(new_img, f'{number}', position, font, font_scale, color, 1, cv2.LINE_AA)
    return im_n

# %% ../nbs/00_core.ipynb 16
def show_poster_from_path(
                mask_path:str, 
                im_path:str, # name of the image fodler , e.g. 'images' or 'X'
                show_:str, # whether to show image, mask, poster
                text:str,# text in image
                scale=1
                ):
    'Show only masked part of the image or full image with mask'
    row_col_dict = {35:(5, 7), 21:(3,7),30:(5,6), 34:(2,17), 22:(2,11), 36:(6,6), 20:(5, 4), 33:(3,11) }

    # getting mask and iamge name from mask
    name = Path(mask_path).name
    im_name = Path(im_path)/name

    msk_ = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img_ = cv2.imread(str(im_name), cv2.IMREAD_GRAYSCALE)
    img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)

    # finding contours and base on that concat only contours
    contrs, _ = cv2.findContours(msk_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row, col = row_col_dict.get(len(contrs), (1,len(contrs)))
    cv2.drawContours(img_, contrs, -1, (0, 255, 0), 1)
    images_list = []
    for c in contrs:
        x, y, w, h = cv2.boundingRect(c)
        offset = 25
        new_img = img_[y-offset:y+h+offset, x-offset:x+w+offset]
        images_list.append(new_img)
    img_new = concat_images(images_list, row, col, number=text)


    # show imagess
    if show_ == 'both':
        print(img_new.shape, img_.shape)
        img_ = cv2.resize(img_, (img_new.shape[1], img_new.shape[0]))
        new_img = cv2.hconcat([img_, img_new])
        res = new_img.shape
        fig, ax = plt.subplots(1, 1, figsize=(scale * res[1]/dpi, scale*res[0]/dpi))
        ax.imshow(new_img)

    elif show_ == 'image':
        res = img_.shape
        fig, ax = plt.subplots(1, 1, figsize=(scale * res[1]/dpi, scale*res[0]/dpi))
        ax.imshow(img_)
    else:
        res = img_new.shape
        fig, ax = plt.subplots(1, 1, figsize=(scale * res[1]/dpi, scale*res[0]/dpi))
        ax.imshow(img_new)

# %% ../nbs/00_core.ipynb 17
def get_template_part(
    img:np.ndarray, #opencv image
    tmp_img:np.ndarray # opencv image
   ):
   'Get bounding box coordinate from the image (x,y, w, h format)'

   res = cv2.matchTemplate(
                           img,
                           tmp_img,
                           cv2.TM_CCOEFF_NORMED
   )

   min_, max_, min_loc, max_loc = cv2.minMaxLoc(res)

   top_left = max_loc
   bottom_right = (top_left[0] + tmp_img.shape[1], top_left[1] + tmp_img.shape[0])

   y = top_left[1]
   h = bottom_right[1] - top_left[1]
   x = top_left[0]
   w = bottom_right[0] - top_left[0]
   return x, y, w, h
   

# %% ../nbs/00_core.ipynb 18
def split_image(
        img:np.ndarray,
        num_splits:int,
        direction:str # horizontal or vertical
        ):
    'Split an image into different parts'

    # Calculate the size of each split
    if direction == 'horizontal':
        split_size = img.shape[1] // num_splits
    elif direction == 'vertical':
        split_size = img.shape[0] // num_splits

    # Split the image and store the parts in a list
    parts = []
    for i in range(num_splits):
        if i == num_splits - 1:  # If this is the last split
            if direction == 'horizontal':
                part = img[:, i*split_size:]
            elif direction == 'vertical':
                part = img[i*split_size:, :]
        else:
            if direction == 'horizontal':
                part = img[:, i*split_size:(i+1)*split_size]
            elif direction == 'vertical':
                part = img[i*split_size:(i+1)*split_size, :]
        parts.append(part)

    return parts

# %% ../nbs/00_core.ipynb 19
def find_contours_binary(
    img:np.ndarray, # binary image 
    ):
    'Return contours from the binary image'
    cntrs, _= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cntrs


# %% ../nbs/00_core.ipynb 21
def foo(): pass
