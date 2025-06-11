import requests
import os
import logging


def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''Import filenames from bucket_folder_url into raw_data_relative_path'''
    os.makedirs(raw_data_relative_path, exist_ok=True)  # Always create if missing
    # download all the files
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if not os.path.isfile(output_file):
            logging.info(f'Downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(input_file)
            if response.status_code == 200:
                with open(output_file, "wb") as text_file:
                    text_file.write(response.content)
            else:
                logging.error(f'Error accessing the object {input_file}: {response.status_code}')
        else:
            logging.info(f'{output_file} already exists, skipping download.')
                
def main(raw_data_relative_path=os.path.join(os.path.dirname(__file__), '../../data/raw_data'), 
        filenames = ["raw.csv"],
        bucket_folder_url= "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"          
        ):
    """ Download data from AWS S3 to ./data/raw_data folder
    """
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('Finished downloading raw data set.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()