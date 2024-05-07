# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_compress_and_filter_data.ipynb.

# %% auto 0
<<<<<<< HEAD
__all__ = ['apply_functions', 'process_image', 'convert_images_to_parquet', 'convert_image_to_parquet_p', 'filter_images']
=======
__all__ = ['apply_functions', 'convert_images_to_parquet', 'filter_images', 'binary_to_image', 'read_images_from_parquet',
           'save_image']
>>>>>>> de4a42b (save image is added in compress_and_filter notebook)

# %% ../nbs/04_compress_and_filter_data.ipynb 3
import os
from PIL import Image
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union, List, Tuple, Callable
from tqdm.auto import tqdm
import os
from fastcore.all import *
from typing import Union, List, Tuple, Callable

# %% ../nbs/04_compress_and_filter_data.ipynb 4
def apply_functions(
        fn: Union[Path, str], # Name fo the file
        functions:Union[List[Callable], None]=None
    ):
    'Apply a list of functions to a file'

    results = {}
    for func in functions:

        try:
            result = func(fn)

            if result is not None:
                results[func.__name__] = result
        except Exception as e:
            print(f'Error in {func.__name__} for {fn}: {e}')
    return results

# %% ../nbs/04_compress_and_filter_data.ipynb 6
def process_image(im, file_name_func):
    fn = im.name
    with Image.open(im) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Apply functions to filename and collect metadata
        if file_name_func is not None:
            metadata = apply_functions(fn, file_name_func)

        data_entry = {'filename': fn, 'image_data': img_byte_arr}
        if file_name_func is not None:
            data_entry.update(metadata)

        return data_entry

# %% ../nbs/04_compress_and_filter_data.ipynb 7
def convert_images_to_parquet(
        image_directory:Union[str, Path],  # directory containing images
        output_file:Union[str, Path],      # output parquet file
        file_name_func:List[Callable]=None, # functions to apply to the filename
        file_exts:str='.png'
        ):
    ' Convert images in a directory to a parquet file '
    images = image_directory.ls(file_exts=file_exts) 
    n_ims = len(images)

    data = []
    for i, im in enumerate(tqdm(images, total=n_ims)):
        fn = im.name
        with Image.open(im) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()


            # Apply functions to filename and collect metadata
            if file_name_func is not None:
                metadata = apply_functions(fn, file_name_func)

            data_entry = {'filename': fn, 'image_data': img_byte_arr}
            if file_name_func is not None:
                data_entry.update(metadata)


            # Append the binary data and filename to the list
            data.append(data_entry)

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)

# %% ../nbs/04_compress_and_filter_data.ipynb 10
def convert_image_to_parquet_p(
    im_path:Union[str, Path], # path of images
    file_name_func:List[Callable]=None, # functions to apply to the filename
    file_exts:str='.png', # file extensions
    threadpool:bool=True, # use threadpool
    num_workers:int=16, # number of workers
    chunksize:int=1, # chunksize
   ):
      ' Convert images in a directory to a parquet file parallell'
      data = parallel(
            f=process_image,
            items=im_path.ls(file_exts=file_exts),
            file_name_func=file_name_func,
            threadpool=threadpool,
            n_workers=num_workers,
            progress=True,
            chunksize=chunksize
      )

      table = pa.Table.from_pandas(pd.DataFrame(data))
      im_path_n = Path(im_path).stem
      pq.write_table(table, Path(im_path.parent, f'{im_path_n}.parquet'))

# %% ../nbs/04_compress_and_filter_data.ipynb 14
def filter_images(parquet_file, condition):
    # Load the Parquet file
    table = pq.read_table(parquet_file)

    # Convert to Pandas DataFrame for easier manipulation
    df = table.to_pandas()

    # Filter the DataFrame based on a condition
    filtered_df = df[df['example_metadata_extractor'].apply(lambda x: condition in x)]

    return filtered_df


# %% ../nbs/04_compress_and_filter_data.ipynb 15
def binary_to_image(data_entry):
    img_data = data_entry['image_data']
    img_byte_arr = io.BytesIO(img_data)
    img = Image.open(img_byte_arr)
    return data_entry['filename'], img

# %% ../nbs/04_compress_and_filter_data.ipynb 16
def read_images_from_parquet(parquet_path, num_workers=16):
    # Read the Parquet file
    table = pq.read_table(parquet_path)
    df = pd.DataFrame(table.to_pandas())
    
    # Use FastAI's parallel function to process images in parallel
    results = parallel(
        binary_to_image, 
        df.to_dict(orient='records'), 
        n_workers=num_workers,
        progress=True

        )
    
    # Convert list of tuples into a dictionary
    images = {filename: img for filename, img in results}
    return images



# %% ../nbs/04_compress_and_filter_data.ipynb 17
# Define a function to save a single image
def save_image(data):
    filename, image, output_dir = data
    # Ensure the image is in the correct format
    if image.mode not in ['L']:
        image = image.convert('L')
    
    # Create the full path for the image
    file_path = Path(output_dir, filename)
    # Save the image
    image.save(file_path, format='PNG')  # Adjust format as needed

