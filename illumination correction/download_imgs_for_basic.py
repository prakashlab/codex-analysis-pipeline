import argparse
import glob
import zarr
import gcsfs
import os
import json
import imageio
import cv2
from tqdm import tqdm

def imread_gcsfs(fs,file_path):
    img_bytes = fs.cat(file_path)
    I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
    return I

if __name__ == '__main__':

    image_format = 'bmp'

    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900

    debug_mode = True

    write_to_gcs = False
    use_zip_store = True

    gcs_project = 'soe-octopi'
    gcs_token = 'uganda-2022-viewing-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_source = 'gs://octopi-malaria-uganda-2022-data'
    bucket_destination = 'gs://octopi-malaria-data-processing'

    fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    # load the list of dataset
    if args.data_id != None:
        DATASET_ID = [args.data_id]
    else:
        f = open('list_of_datasets_uganda_2022.txt','r')
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
        f.close()

    # go through dataset
    for dataset_id in DATASET_ID:

        print(dataset_id)

        json_file = fs.cat(bucket_source + '/' + dataset_id + '/acquisition parameters.json')
        acquisition_parameters = json.loads(json_file)

        parameters['row_start'] = 0
        parameters['row_end'] = acquisition_parameters['Ny']
        parameters['column_start'] = 0
        parameters['column_end'] = acquisition_parameters['Nx']
        parameters['z_start'] = 0
        parameters['z_end'] = acquisition_parameters['Nz']
        if debug_mode:
            parameters['row_start'] = 13-5
            parameters['row_end'] = 13+5
            parameters['column_start'] = 13-5
            parameters['column_end'] = 13+5

        dir_out = dataset_id
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        os.mkdir(dir_out + '/R')
        os.mkdir(dir_out + '/G')
        os.mkdir(dir_out + '/B')
        os.mkdir(dir_out + '/left')
        os.mkdir(dir_out + '/right')

        for i in tqdm(range(parameters['row_start'],parameters['row_end'])):
            for j in range(parameters['column_start'],parameters['column_end']):
                for k in range(parameters['z_start'],parameters['z_end']):
                    file_id = str(i) + '_' + str(j) + '_' + str(k)
                    print('processing fov ' + file_id)

                    I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
                    I_fluorescence = cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dir_out + '/B/' + str(i) + '_' + str(j) + '.' + image_format, I_fluorescence[:,:,0])
                    cv2.imwrite(dir_out + '/G/' + str(i) + '_' + str(j) + '.' + image_format, I_fluorescence[:,:,1])
                    cv2.imwrite(dir_out + '/R/' + str(i) + '_' + str(j) + '.' + image_format, I_fluorescence[:,:,2])

                    
                    I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
                    I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
                    if I_BF_left.ndim==3:
                        cv2.imwrite(dir_out + '/left/' + str(i) + '_' + str(j) + '.' + image_format, I_BF_left[:,:,1])
                        cv2.imwrite(dir_out + '/right/' + str(i) + '_' + str(j) + '.' + image_format, I_BF_right[:,:,1])
                    else:
                        cv2.imwrite(dir_out + '/left/' + str(i) + '_' + str(j) + '.' + image_format, I_BF_left)
                        cv2.imwrite(dir_out + '/right/' + str(i) + '_' + str(j) + '.' + image_format, I_BF_right)
                    print(dir_out + '/B/')
                    
