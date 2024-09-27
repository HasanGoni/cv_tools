"""Data reading, writing or Saving in S3 bucket with boto3"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/08_data_processing.s3_bucket.ipynb.

# %% auto 0
__all__ = ['CV_TOOLS', 'custom_lib_path', 's3_access_key_id', 's3_secret_access_key', 'CURRETNT_NB', 'get_client',
           'list_s3_folder_contents', 'ls_s3', 'download_s3_folder', 'upload_to_s3', 'upload_download_s3']

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 3
import sys
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# %% ../../nbs/08_data_processing.s3_bucket.ipynb 4
import os
import boto3
from nbdev.showdoc import show_doc
from botocore.exceptions import ClientError

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 5
CV_TOOLS = Path(r'/home/ai_sintercra/homes/hasan/projects/git_data/cv_tools')
sys.path.append(str(CV_TOOLS))


# %% ../../nbs/08_data_processing.s3_bucket.ipynb 6
custom_lib_path = Path(r'/home/ai_warstein/homes/goni/custom_libs')
sys.path.append(str(custom_lib_path))


# %% ../../nbs/08_data_processing.s3_bucket.ipynb 7
from ..imports import *
from dotenv import load_dotenv


# %% ../../nbs/08_data_processing.s3_bucket.ipynb 8
load_dotenv(dotenv_path=f'/home/ai_sintercra/homes/hasan/projects/git_data/2023_easy_pin_detection/private_easy_pin_detection/.env')

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 10
s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 11
CURRETNT_NB='/home/ai_sintercra/homes/hasan/projects/git_data/2023_easy_pin_detection/nbs'

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 12
def get_client(
    s3_access_key_id:str, 
    s3_secret_access_key:str,
    endpoint_url:str='https://s3warceph01.infineon.com',
    verify:bool=False):
    return boto3.client(
        's3', 
        endpoint_url=endpoint_url, 
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        verify=verify,
        )
        
       

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 15
def list_s3_folder_contents(
    boto3_client: boto3.client, 
    bucket_name: str, 
    folder_prefix: str,
    recursive: bool = False) -> List[str]:
    """
    Make sure to add '/' at the end of the folder_prefix
    List files and subfolders in a specific S3 bucket folder.
    If recursive is True, it will list all files and subfolders recursively.
    If recursive is False, it will only list immediate contents of the folder.
    
    """
    contents = set()
    try:
        paginator = boto3_client.get_paginator('list_objects_v2')
        
        if recursive:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)
        else:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix, Delimiter='/')
        
        for page in pages:
            if not recursive and 'CommonPrefixes' in page:
            
                for prefix in page['CommonPrefixes']:
                    contents.add(prefix['Prefix'])
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key != folder_prefix:  # Exclude the folder itself
                        if recursive:
                            contents.add(key)
                        else:
                            # Only add files in the immediate folder
                            relative_path = key[len(folder_prefix):]
                            if not relative_path.strip('/'):  # This is the folder itself
                                continue
                            if '/' not in relative_path.strip('/'):
                                contents.add(relative_path.split('/')[0])
        
        contents_list = sorted(list(contents))
        if contents_list:
            print(f"Found {len(contents_list)} items in the folder '{folder_prefix}':")
            #for item in contents_list[:10]:  # Print first 10 items
                #print(item)
                
            if len(contents_list) > 10:
                print("...")
        else:
            print(f"No items found in the folder '{folder_prefix}'.")
        
        return contents_list
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []



# %% ../../nbs/08_data_processing.s3_bucket.ipynb 17
def ls_s3(
    folder_prefix: str,
    bucket_name: str='s3-ai-warstein', 
    s3_access_key_id: str=os.environ['AWS_ACCESS_KEY_ID'],
    s3_secret_access_key: str=os.environ['AWS_SECRET_ACCESS_KEY'],
    recursive: bool=False,
    endpoint_url: str='https://s3warceph01.infineon.com',
    verify: bool=False,
    ) -> List[str]:
    boto3_client = get_client(
        s3_access_key_id=s3_access_key_id, 
        s3_secret_access_key=s3_secret_access_key, 
        endpoint_url=endpoint_url,
        verify=verify)
    #return None
    return list_s3_folder_contents(
           boto3_client=boto3_client, 
           bucket_name=bucket_name, 
           folder_prefix=folder_prefix,
           recursive=recursive)

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 21
def download_s3_folder(
    s3_folder: str,
    local_dir: str,
    bucket_name: str = 's3-ai-warstein',
    s3_access_key_id: str = os.environ['AWS_ACCESS_KEY_ID'],
    s3_secret_access_key: str = os.environ['AWS_SECRET_ACCESS_KEY'],
    endpoint_url: str = 'https://s3warceph01.infineon.com',
    verify: bool = False,
    file_num: int = None
):
    """
    Download all contents of an S3 folder to a local directory.

    Args:
    s3_folder (str): The S3 folder path to download from.
    local_dir (str): The local directory to save the downloaded files.
    bucket_name (str): The S3 bucket name.
    s3_access_key_id (str): AWS access key ID.
    s3_secret_access_key (str): AWS secret access key.
    endpoint_url (str): S3 endpoint URL.
    verify (bool): Whether to verify SSL certificates.
    """
    s3_client = get_client(
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        endpoint_url=endpoint_url,
        verify=verify
    )

    # Ensure the S3 folder path ends with a '/'
    if not s3_folder.endswith('/'):
        s3_folder += '/'

    # List all objects in the S3 folder
    objects = ls_s3(
        folder_prefix=s3_folder,
        bucket_name=bucket_name,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        recursive=True
    )
    if file_num is not None:
        objects = objects[:file_num]

    for obj in objects:
        # Get the relative path of the file
        relative_path = obj[len(s3_folder):]
        # Construct the full local path
        local_file_path = os.path.join(local_dir, relative_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        try:
            s3_client.download_file(bucket_name, obj, local_file_path)
            print(f"Downloaded: {obj} to {local_file_path}")
        except Exception as e:
            print(f"Error downloading {obj}: {str(e)}")

    print(f"Download complete. All files from {s3_folder} have been downloaded to {local_dir}")


# %% ../../nbs/08_data_processing.s3_bucket.ipynb 23
def upload_to_s3(
    local_path, 
    s3_folder, 
    bucket_name, 
    s3_access_key_id, 
    s3_secret_access_key,
    endpoint_url: str = 'https://s3warceph01.infineon.com',
    verify: bool = False,
    ):
    """
    Upload a file or folder to S3 bucket. If the folder doesn't exist, it will be created.
    
    :param local_path: Path to the local file or directory to upload
    :param s3_folder: S3 folder path where the file/folder will be uploaded
    :param bucket_name: Name of the S3 bucket
    :param s3_access_key_id: AWS access key ID
    :param s3_secret_access_key: AWS secret access key
    """
    # Create S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        endpoint_url=endpoint_url,
        verify=verify
    )

    # Ensure the S3 folder path ends with a '/'
    if not s3_folder.endswith('/'):
        s3_folder += '/'

    if os.path.isfile(local_path):
        # Upload single file
        file_name = Path(local_path).name
        s3_path = s3_folder + file_name
        try:
            s3_client.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded: {local_path} to {s3_path}")
        except Exception as e:
            print(f"Error uploading {local_path}: {str(e)}")
    elif Path(local_path).is_dir():
        # Upload entire folder
        for i in tqdm(Path(local_path).ls(),total=len(Path(local_path).ls())):
            if i.is_file():
                s3_path = s3_folder + i.name
                try:
                    s3_client.upload_file(str(i), bucket_name, s3_path)
                    print(f"Uploaded: {i} to {s3_path}")
                except Exception as e:
                    print(f"Error uploading {i}: {str(e)}")
            elif i.is_dir():
                upload_to_s3(str(i), s3_folder + i.name, bucket_name, s3_access_key_id, s3_secret_access_key)
    else:
        print(f"Error: {local_path} is not a valid file or directory")

    print(f"Upload complete. All files from {local_path} have been uploaded to {s3_folder} in bucket {bucket_name}")

# %% ../../nbs/08_data_processing.s3_bucket.ipynb 24
@call_parse
def upload_download_s3(
    download:Param(help='whether to download or upload',type=bool, action='store_true' ),
    verify:Param(help='whether to verify ssl certificates',type=bool, action='store_false'),
    local_path:Param(help='local path to the file or folder to upload',type=str)='test',
    s3_folder:Param(help='s3 folder path where the file or folder will be uploaded',type=str)='s_test',
    bucket_name:Param(help='name of the s3 bucket',type=str)='s3-ai-warstein',
):
    """
    Upload a file or folder to S3 bucket. If the folder doesn't exist, it will be created.
    
    :param local_path: Path to the local file or directory to upload
    :param s3_folder: S3 folder path where the file/folder will be uploaded
    :param bucket_name: Name of the S3 bucket
    :param s3_access_key_id: AWS access key ID
    :param s3_secret_access_key: AWS secret access key
    """
    # load env variables
    load_dotenv(dotenv_path=f'/home/ai_sintercra/homes/hasan/projects/git_data/2023_easy_pin_detection/private_easy_pin_detection/.env')

    # get env variables
    s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    endpoint_url = 'https://s3warceph01.infineon.com'


    if download:
        download_s3_folder(
            s3_folder=s3_folder,
            local_dir=local_path,
            bucket_name=bucket_name,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            endpoint_url=endpoint_url,
            verify=verify
        )
    else:
        upload_to_s3(
            local_path=local_path,
            s3_folder=s3_folder,
            bucket_name=bucket_name,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            endpoint_url=endpoint_url,
            verify=verify
        )

    