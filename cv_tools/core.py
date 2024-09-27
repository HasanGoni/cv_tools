"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['OpenCvImage', 'get_name_', 'dpi', 'read_config', 'label_mask', 'find_rotation_angle', 'rotate_image',
           'show_labeled_mask', 'write_new_mask', 'remove_object_from_mask', 'read_img', 'blur_img_', 'median_img_',
           'canny_img_', 'sharpen_img_', 'bilateral_img_', 'nlm_filter_', 'tv_filter_', 'filter_img_',
           'add_text_to_image', 'denoise_img', 'show_', 'center_crop', 'overlay_mask', 'overlay_mask_border_on_image',
           'overlay_mask_border_on_image_frm_img', 'concat_images', 'show_poster_from_path', 'seamless_clone',
           'get_template_part', 'split_image', 'split_image_with_coordinates', 'create_same_shape',
           'get_circle_from_single_pin', 'find_contours_binary', 'adjust_brightness', 'ssim_', 'orb_sim_',
           'rot_based_on_ref_img', 'frm_cntr_to_bbox', 'foo']

# %% ../nbs/00_core.ipynb 3
from PIL import Image
import cv2
from fastcore.all import *
from pathlib import Path
import numpy as np 
import pandas as pd
import shutil
from tqdm.auto import tqdm
from typing import Union, Dict, List, NewType
import matplotlib.pyplot as plt
from typing import Tuple
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import (
    label, binary_dilation, binary_erosion,label,
    )
from skimage.color import label2rgb
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
import yaml
from skimage import img_as_float
import matplotlib as mpl

# %% ../nbs/00_core.ipynb 4
OpenCvImage = NewType('OpenCvImage', np.ndarray)

# %% ../nbs/00_core.ipynb 5
get_name_ = np.vectorize(lambda x: x.name)
dpi = mpl.rcParams['figure.dpi']

# %% ../nbs/00_core.ipynb 7
@patch
def filter_(self:Path, name_part:str):
    'filter based on name_part in file' 
    return  L(filter(lambda x: name_part in x.name, self.ls()))

# %% ../nbs/00_core.ipynb 8
def read_config(config_file:Path) -> Dict:
    'read config file'
    with open (f'{config_file}', 'r') as file:
        config = yaml.safe_load(file)
    return config

# %% ../nbs/00_core.ipynb 10
def label_mask(
        mask:np.ndarray
        )->np.ndarray:
    "Label connected components in a binary mask"
    try: 
        labels, num_labels = label(mask)
        return labels, num_labels
    except Exception as e:
        print(e)


# %% ../nbs/00_core.ipynb 11
def find_rotation_angle(
    ref_img:np.ndarray, 
    tst_img:np.ndarray,
    method:str='sift' # sift or orb
    )->float:
    'Find rotation angle between ref image and test image'
    if method == 'sift':
        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(ref_img, None)
        kp2, des2 = sift.detectAndCompute(tst_img, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # FLANN-based matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    else:
        method = 'orb'
        detector = cv2.ORB_create()
        kp1, des1 = detector.detectAndCompute(ref_img, None)
        kp2, des2 = detector.detectAndCompute(tst_img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        good_matches = matches[:10]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)



    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Extract rotation angle from homography matrix
    angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

    return angle



# %% ../nbs/00_core.ipynb 12
def rotate_image(
    image:Union[OpenCvImage,Image.Image], # image 
    angle:int, # angle to rotate
    interpolation_method=cv2.INTER_CUBIC
    )-> OpenCvImage: # return only rotated image
    ' Rotate image by angle'
     # if Pil image
    if isinstance(image, Image.Image):
        image = np.array(image, dtype=np.uint8)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_image = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=interpolation_method,
        borderMode=cv2.BORDER_REPLICATE)
    return rot_image


# %% ../nbs/00_core.ipynb 13
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

# %% ../nbs/00_core.ipynb 14
def write_new_mask(
                     new_lb, # labeled masks
                     new_mask_path, # path to save the new mask
                     fn # name of the image
                     ):
    'Write mask to new_mask_path'
    new_lbl = new_lb.astype(np.uint8)
    new_lbl = np.where(new_lbl > .9, 255, 0)
    cv2.imwrite(f'{new_mask_path}/{fn}', new_lbl)

# %% ../nbs/00_core.ipynb 15
def remove_object_from_mask(
                           mask:np.ndarray,
                           object_id_list:List[int]
                           ):
    """Remove object from mask."""

    for i in object_id_list:
        mask[mask == i] = 0
    return mask

# %% ../nbs/00_core.ipynb 17
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
    
    else:
        img = Image.open(im_path)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')

# %% ../nbs/00_core.ipynb 18
def blur_img_(
        img:np.ndarray,  #image
        ks:int=3, # kernel size
        ) -> np.ndarray:
    'Blur image'
    return cv2.GaussianBlur(img, (ks, ks), 0)


# %% ../nbs/00_core.ipynb 19
def median_img_(
        img:np.ndarray,  #image
        ks:int=3, # kernel size
        ) -> np.ndarray:
    'Median image'
    return cv2.medianBlur(img, ks)


# %% ../nbs/00_core.ipynb 20
def canny_img_(
        img:np.ndarray,  #image
        th1:int=50, #threshold1
        th2:int=150, #threshold2
        ) -> np.ndarray:
    'Canny image'
    return cv2.Canny(img, th1, th2)


# %% ../nbs/00_core.ipynb 21
def sharpen_img_(
        img:np.ndarray,  #image
        ) -> np.ndarray:
    'Sharpen image'
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)




# %% ../nbs/00_core.ipynb 22
def bilateral_img_(
        img:np.ndarray,  #image
        d:int=9, # diameter of each pixel neighborhood
        sc:int=75, # sigma color
        ss:int=75, # sigma space
        ) -> np.ndarray:
    'Bilateral image'
    return cv2.bilateralFilter(img, d, sc, ss)

# %% ../nbs/00_core.ipynb 23
def nlm_filter_(
    img:np.ndarray,  #image
    h:float=1.15,
    fast_mode:bool=True,
    patch_size:int=5,
    patch_distance:int=3

  ):
  'Apply Non local means denoising filter to image'
  img = img.astype(np.float)
  sigma_est = np.mean(
    estimate_sigma(img) 
    )
  denoise_img = denoise_nl_means(
    img, 
    h=h* sigma_est, 
    fast_mode=fast_mode,
    patch_size=patch_size,
    patch_distance=patch_distance,
    )
  return denoise_img

# %% ../nbs/00_core.ipynb 24
def tv_filter_(
    img:np.ndarray,  #image
    weight:float=0.1, # greater the weight, more denoising, but with less details
    max_num_iter:int=200,

    ):
    'Apply total variation denoising filter to image'
    #img = img.astype(np.float)
    #d_img = denoise_tv_chambolle(
        #img, 
        #weight=0.1, 
        #eps=0.0002,
        #max_num_iter=max_num_iter,
        #)
    dst = cv2.fastNlMeansDenoising(np.clip(img,0,255), None, 4, 7, 35)
    return dst
    

# %% ../nbs/00_core.ipynb 26
def filter_img_(
        img:np.ndarray,  #image
        filters_:List[str] # opencv filters
        ) -> np.ndarray:

    'Apply filter on image'

    ops = {
        'blur': blur_img_,
        'med': median_img_,
        'canny': canny_img_,
        'sharpen': sharpen_img_,
        'nlm': nlm_filter_,
        'tv': tv_filter_,
        'bilateral': bilateral_img_,
    }
    for fl in filters_:
        if fl in ops:
            img = ops[fl](img)
        else:
            print(f'Filter {fl} not found')
    return img

# %% ../nbs/00_core.ipynb 27
def add_text_to_image(
                        img: np.ndarray, 
                        text: str, 
                        position: Tuple[int, int]=None,
                        font=cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale: float=1, 
                        color: Tuple[int, int, int]=(255, 255, 255), 
                        thickness: int=2):
    """
    Add text to an image.

    Parameters:
    img (np.ndarray): The image to add text to.
    text (str): The text to add.
    position (Tuple[int, int]): The position where the text should be added.
    font: The font to use. Default is cv2.FONT_HERSHEY_SIMPLEX.
    font_scale (float): The font scale. Default is 1.
    color (Tuple[int, int, int]): The color of the text in BGR format. Default is white.
    thickness (int): The thickness of the text. Default is 2.

    Returns:
    np.ndarray: The image with the text added.
    """
    if position is None:
        # Set the position to a small offset from the top left corner
        position = (10, 30)  # 10 pixels from the left, 30 pixels from the top
    img_with_text = cv2.putText(img.copy(), text, position, font, font_scale, color, thickness)
    return img_with_text

# %% ../nbs/00_core.ipynb 28
def denoise_img(
    img:np.ndarray,  #image
    filters_:List[str], # opencv filters
    h:float=1.15,
    fast_mode:bool=True,
    patch_size:int=5,
    patch_distance:int=3,
    multichannel:bool=False,
    weight:float=0.1, # greater the weight, more denoising, but with less details
    n_iter_max:int=200,
    ):
    'Apply filter on image'

    ops = {
        'blur': blur_img_,
        'med': median_img_,
        'canny': canny_img_,
        'sharpen': sharpen_img_,
        'nlm': nlm_filter_,
        'tv': tv_filter_,
        'bilateral': bilateral_img_,
    }
    images_= []
    for fl in filters_:
        if fl in ops:
            img_ = ops[fl](img)
            image_ = add_text_to_image(img_, fl)
            images_.append(image_)

        else:
            print(f'Filter {fl} not found')
    return show_(np.concatenate(images_, axis=1))

# %% ../nbs/00_core.ipynb 29
def show_(
    im_path:Union[str, np.ndarray],
	cmap:str='gray'
    ):
    'Showing an image could be image or str, '

    if isinstance(im_path,str) or isinstance(im_path, Path):
        im = Image.open(im_path)
        fig, ax = plt.subplots(figsize= (im.size[0]/dpi, im.size[1]/dpi))
        ax.imshow(im, cmap=cmap)
        ax.axis('off')
    elif isinstance(im_path, list):
            for i in im_path:
                if isinstance(i, str) or isinstance(i, Path):
                    im = Image.open(i)
                    h, w = im.size
                    fig, ax = plt.subplots(figsize= (w/dpi, h/dpi))
                    ax.imshow(i, cmap=cmap)
                    ax.axis('off')
                elif isinstance(i, np.ndarray):
                    h, w = i.shape[0], i.shape[1]
                    fig, ax = plt.subplots(figsize= (w/dpi, h/dpi))
                    ax.imshow(i, cmap=cmap)
                    ax.axis('off')
                    
                    
    else: 
        h, w = im_path.shape[0], im_path.shape[1]
        fig, ax = plt.subplots(figsize= (w/dpi, h/dpi))
        im = im_path
        ax.imshow(im,cmap=cmap)
        ax.axis('off')

# %% ../nbs/00_core.ipynb 30
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

# %% ../nbs/00_core.ipynb 31
def center_crop(
        image:Image,# not open cv but PIL Image 
        desired_width:int=1632,
        desired_height:int=1152,
        height_offset:int=-50,
        width_offset:int=-70,
        cv:bool=True
        ):
    # Get the current size of the image
    width, height = image.size

    # Calculate the coordinates of the center of the image
    center_x = (width // 2 + (width_offset))
    center_y = (height // 2 + (height_offset))

    # Calculate the coordinates of the top-left corner of the crop
    left = center_x - desired_width // 2
    top = center_y - desired_height // 2

    # Calculate the coordinates of the bottom-right corner of the crop
    right = center_x + desired_width // 2
    bottom = center_y + desired_height // 2

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    #print(f'cropped_image size = {cropped_image.size}')
    if cv:
        return np.array(cropped_image)

    return cropped_image

# %% ../nbs/00_core.ipynb 32
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

# %% ../nbs/00_core.ipynb 33
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



# %% ../nbs/00_core.ipynb 34
def overlay_mask_border_on_image_frm_img(
        img:Union[OpenCvImage, Image.Image],
        msk:Union[OpenCvImage, Image.Image],
    ):
    'Overlay mask border on image from image'
    cv2_img = np.array(img) if isinstance(img, Image.Image) else img
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    # converting cv2 mask in case it is not 
    cv2_msk = np.array(msk) if isinstance(msk, Image.Image) else msk

    msk_uint8 = (cv2_msk* 255).astype(np.uint8)
    if len(msk_uint8.shape) == 3 and msk_uint8.shape[2] == 3:
        msk_uint8 = cv2.cvtColor(msk_uint8, cv2.COLOR_RGB2GRAY)
    cntrs, _, = cv2.findContours(msk_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
        cv2_img, 
        contours=cntrs, 
        contourIdx=-1, 
        color=(0, 255, 0), 
        thickness=2)
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)


# %% ../nbs/00_core.ipynb 35
def concat_images(
        images:List[np.ndarray],
        rows:int,  # number of rows
        cols:int,  # number of columns in combined images
        number:str, # a text which will be inserted in the combined image
        border_size:int=10, # border size
        border_color:Tuple[int, int, int]=(0, 0, 0) # border color
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
        row_images = [cv2.copyMakeBorder(img, 0, 0, 0, border_size, cv2.BORDER_CONSTANT, value=border_color) for img in row_images]
        row_concat = cv2.hconcat(row_images)

        res.append(row_concat)
    new_img = cv2.vconcat(res)
    position = (6, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale=0.6
    color = (255, 0, 0)
    im_n = cv2.putText(new_img, f'{number}', position, font, font_scale, color, 1, cv2.LINE_AA)
    return im_n

# %% ../nbs/00_core.ipynb 36
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

# %% ../nbs/00_core.ipynb 37
def seamless_clone(
        full_img:OpenCvImage, # gray scale image
        replace_part:OpenCvImage, # gray scale image
        center:Tuple[int, int],# center position in full_img #where the replace_part will be placed
        clone_method:str='normal'
    ):
    'Replace some part of full_img with replace_part object'
    full_img = cv2.cvtColor(full_img, cv2.COLOR_GRAY2BGR)

    replace_part = cv2.cvtColor(replace_part, cv2.COLOR_GRAY2BGR)
    mask = np.ones_like(replace_part)*255
    if clone_method == 'mixed':
        cloned_img = cv2.seamlessClone(
            replace_part,
            full_img,
            mask,
            center,
            cv2.MIXED_CLONE)
    else:
        cloned_img = cv2.seamlessClone(
            replace_part,
            full_img,
            mask,
            center,
            cv2.NORMAL_CLONE)
    
    return cv2.cvtColor(cloned_img, cv2.COLOR_BGR2GRAY)

# %% ../nbs/00_core.ipynb 38
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
   

# %% ../nbs/00_core.ipynb 39
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

# %% ../nbs/00_core.ipynb 40
def split_image_with_coordinates(
        img: np.ndarray, 
        num_splits: int, 
        direction: str) -> list:
    'Split an image with different parts in different coordinates'
    indexed_parts_with_coords = []
    split_size = img.shape[0] // num_splits if direction == 'vertical' else img.shape[1] // num_splits

    for i in range(num_splits):
        if direction == 'vertical':
            start_y = i * split_size
            end_y = img.shape[0] if i == num_splits - 1 else (i + 1) * split_size
            part = img[start_y:end_y, :]
            coordinates = (0, start_y, img.shape[1], end_y)
        else:  # 'horizontal'
            start_x = i * split_size
            end_x = img.shape[1] if i == num_splits - 1 else (i + 1) * split_size
            part = img[:, start_x:end_x]
            coordinates = (start_x, 0, end_x, img.shape[0])
        indexed_parts_with_coords.append((i, part, coordinates))

    return indexed_parts_with_coords

# %% ../nbs/00_core.ipynb 41
def create_same_shape(
        src_img:np.ndarray, # image which size needs to be replicated
        dst_img:np.ndarray, # image which needs to resized
    ):
    'Create same shape of dst image like src_img'

    h, w = src_img.shape[:2]
    add_h, add_w = dst_img.shape[:2]

    if h > add_h or w > add_w:
        return cv2.resize(dst_img, (w, h))
    else:
        start_h, start_w = (add_h - h) // 2, (add_w - w) // 2
        end_h, end_w = start_h + h, start_w + w
        new = dst_img[start_h:end_h, start_w:end_w]
        return new


# %% ../nbs/00_core.ipynb 42
def get_circle_from_single_pin(
                            sn_pin_img:np.ndarray
                              )->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    'Get the circle from single pin image'

    # Apply Gaussian blur to reduce noise
    pin_img = cv2.GaussianBlur(sn_pin_img, (5, 5), 0)


    ## Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
                               pin_img, 
                               cv2.HOUGH_GRADIENT, 
                               1, 
                               20,
                               param1=50, 
                               param2=30, 
                               minRadius=0, 
                               maxRadius=0)
    
    # If circles are detected
    if circles is not None:
        # Get the coordinates and radius of the detected circle
        circles = np.uint16(np.around(circles))

        mask = np.zeros_like(pin_img)  # Mask image with same dimensions as original, initialized to black
        rest = np.copy(pin_img)  # Copy of the original image to isolate the non-circular part
        print(f' Number of circles found  = {len(circles[0])}')

        for i in circles[0, :]:
            # Draw the outer circle on the mask and fill it to create a solid circle
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)  # White circle on black background

            # Cut the circular part from the rest image
            cv2.circle(rest, (i[0], i[1]), i[2], (0, 0, 0), thickness=-1)  # Draw black circle on original

        # Apply mask to the original image to extract only the circular part
        segmented_circle = cv2.bitwise_and(pin_img, mask)
        return segmented_circle, mask, rest
    else:
        print('No circles found')
        return None, None, None


# %% ../nbs/00_core.ipynb 43
def find_contours_binary(
    img:np.ndarray, # binary image 
    ):
    'Return contours from the binary image'
    cntrs, _= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cntrs


# %% ../nbs/00_core.ipynb 44
def adjust_brightness(
    img:np.ndarray, # image to adjust brightness
    alpha:float, # alpha > 1 to brighten; alpha < 1 to darken
    ):
    'Adjust the brightness of the image'
    adjusted = cv2.convertScaleAbs(
        img, 
        alpha=alpha)  # alpha > 1 to brighten; alpha < 1 to darken
    return adjusted

# %% ../nbs/00_core.ipynb 45
def ssim_(
        img1:np.ndarray, 
        img2:np.ndarray,
        win_size:int=5
        ):
    'Compare structural similarity between two images'
    return ssim(img1, img2, win_size=win_size
                )

# %% ../nbs/00_core.ipynb 46
def orb_sim_(
        img1:np.ndarray, 
        img2:np.ndarray):

    'Compare ORB similarity between two images'
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    # finding how many regions are there

    sim_reg = list(filter(lambda x: x.distance < 50, matches))
    if len(matches) == 0:
        return 0
    else:
        return len(list(sim_reg))/len(matches)


# %% ../nbs/00_core.ipynb 47
def rot_based_on_ref_img(
    ref_img:Union[np.ndarray, Image.Image], # Image will be used refeernece image
    tst_img:Union[np.ndarray, Image.Image], # Image where searched for key points
    show_m:bool=True # whether to show matching points of two images
    )->np.ndarray:
    'Rotate the tst_img based on ref image, find key points in ref image and then rotate tst_img'
    if isinstance(ref_img, Image.Image):
        ref_img = np.array(ref_img)
    
    if isinstance(tst_img, Image.Image):
        tst_img = np.array(tst_img)
    orb = cv2.ORB_create(50)
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(tst_img, None)  #kp1 --> list of keypoints
    kp2, des2 = orb.detectAndCompute(ref_img, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    #Match descriptors.
    matches = matcher.match(des1, des2, None) 
    #Creates a list of all matches, just like keypoints
    img3 = cv2.drawMatches(tst_img,kp1, ref_img, kp2, matches[:10], None)
    if show_m: show_(img3)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)  
    #Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if len(tst_img.shape) > 2:
        height, width, ch = ref_img.shape
    else:
        height, width = ref_img.shape
    
    im1Reg = cv2.warpPerspective(tst_img, h, (width, height)) 
    return im1Reg
    


# %% ../nbs/00_core.ipynb 48
def frm_cntr_to_bbox(cntr):
    x,y,w,h = cv2.boundingRect(cntr)
    return x,y,w,h

# %% ../nbs/00_core.ipynb 49
def foo(): pass
