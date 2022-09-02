import numpy as np
import pyvips # https://www.libvips.org/install.html
import gcsfs
import os
import shutil
import time
import xarray as xr
import imageio
from natsort import natsorted
import glob
import cv2

def main():
    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900
    make_viewer = True
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    gcs_project = 'soe-octopi'
    src = "/media/prakashlab/T7/"
    dst = '/home/prakashlab/Documents/kmarx/openseadragon/disp'
    exp_id = "20220823_20x_PBMC_2"
    cha = ["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex", "Fluorescence_638_nm_Ex"]
    cy = [1]
    zstack = 'f'
    make_zoom(parameters, make_viewer, key, gcs_project, src, dst, exp_id, cha, cy, zstack)

def make_zoom(parameters, make_viewer, key, gcs_project, src, dst, exp_id, cha, cy, k):
    t0 = time.time()

    # check if source/dest is remote/local
    root_remote = False
    if src[0:5] == 'gs://':
        root_remote = True
    out_remote = False
    if dst[0:5] == 'gs://':
        out_remote = True
    fs = None
    if root_remote or out_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    # only get the given z slice but get all i, j, ch
    path = src + exp_id + "**/0/**_" + k + "_**.png"
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True) if p.split('/')[-2] == '0']
    # remove duplicates
    imgpaths = list(dict.fromkeys(allpaths))
    imgpaths = np.array(natsorted(imgpaths))
    cycle_names = [i.split('/')[-3] for i in imgpaths]

    parameters['row_start'] = 0
    parameters['row_end'] = int(imgpaths[-1].split('/')[-1].split('_')[0])
    parameters['column_start'] = 0
    parameters['column_end'] = int(imgpaths[-1].split('/')[-1].split('_')[1])
    w = parameters['column_end'] - parameters['column_start']
    h = parameters['row_end'] - parameters['row_start']

    dests = []
    for c in cy:
        for channel in cha:
            # vimgs
            vimgs_I = []
            break_flag = False
            # go through the scan
            for i in range(parameters['row_end']-1,parameters['row_start']-1,-1):
                if break_flag:
                    break
                for j in range(parameters['column_start'],parameters['column_end']):
                    if break_flag:
                        break
                    file_id = str(i) + '_' + str(j) + '_' + k + '_' + channel + '.png'
                    im_path = src + exp_id + '/' + cycle_names[c] + '/0/' + file_id
                    print('processing fov ' + im_path)

                    try:
                        if root_remote:
                            I = imread_gcsfs(fs, im_path)
                        else:
                            I = cv2.imread(im_path)
                    except FileNotFoundError:
                        print(file_id + " not found. This dataset will be skipped.")
                        break_flag = True
                        break
                    
                    # convert to mono if color
                    if len(I.shape)==3:
                        I = I[:,:,0] 
                                       
                    # normalize and subtract min
                    I = I - np.min(I)
                    I = I.astype('float') / np.max(I)
                    I = I * 255
                    
                    # crop image
                    I = I[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]

                    # vips
                    tmp_I = pyvips.Image.new_temp_file(str(i)+'_'+str(j)+ '_' + channel +'.v')
                    tmp = pyvips.Image.new_from_array(I.astype('uint8'))
                    tmp.write(tmp_I)
                    vimgs_I.append(tmp_I)

            if break_flag:
                continue

            print('joining arrays')
            vimgs_I = pyvips.Image.arrayjoin(vimgs_I, across=w)

            print('writing to files')
            # output dir
            fname =  cycle_names[c] + '_' + k + '_' + channel + '.png'
            savepath = dst + exp_id + 'deepzooms/'
            print(savepath+fname)
            if out_remote:
                vimgs_I.dzsave(fname, tile_size=1024, suffix='.jpg[Q=95]')
                fs.put(fname+'.dzi', savepath+fname+'.dzi')
                fs.put(fname + "_files", savepath + fname + "_files", recursive=True)
                os.remove(fname + '.dzi')
                shutil.rmtree(fname + "_files")
            else:
                os.makedirs(savepath, exist_ok=True)
                vimgs_I.dzsave(savepath+fname, tile_size=1024, suffix='.jpg[Q=95]')
            if make_viewer:
                dest = savepath+fname
                if out_remote:
                    # strip "gs:/"
                    dest = dest[4:]
                dests.append(dest) 

    # optionally generate a viewer
    if make_viewer:
        filesave = dst
        if out_remote:
            filesave = './temp/'
        with open(filesave + exp_id + "_all_viewer.html") as f:
            f.write('''<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <style>
      #seadragon-viewer {
        position: absolute;
        left: 0;
        top: 90px;
        right: 0;
        bottom: 0;
      }
    </style>
  </head>
  <body>
  <div class="output"></div>
    <div id="seadragon-viewer"></div>
    <script
      src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"
      type="text/javascript"
    ></script>
    <script
      src="openseadragon/openseadragon.min.js"
      type="text/javascript"
    ></script>
    <script type="text/javascript">
      paths =
  ''')
            f.write(str(dests))
            f.write(''';
      function load_view(key) {
        try {
          $("#seadragon-viewer").children().remove();
        } catch (err) {
          console.log(err.message);
        }
        var viewer = OpenSeadragon({
          id: "seadragon-viewer",
          prefixUrl: "openseadragon/images/",
          tileSources: paths,
          collectionMode: true,
          gestureSettingsMouse: { clickToZoom: false },
          collectionRows: 6,
          collectionTileMargin: 16,
        });
        var hitTest = function (position) {
          var box;
          var count = viewer.world.getItemCount();
          for (var i = 0; i < count; i++) {
            box = viewer.world.getItemAt(i).getBounds();
            if (
              position.x > box.x &&
              position.y > box.y &&
              position.x < box.x + box.width &&
              position.y < box.y + box.height
            ) {
              return i;
            }
          }

          return -1;
        };

        var $viewerElement = $("#seadragon-viewer").on(
          "mousemove",
          function (event) {
            var pixel = new OpenSeadragon.Point(event.clientX, event.clientY);
            pixel.y -= $viewerElement.position().top;
            var index = hitTest(viewer.viewport.pointFromPixel(pixel));
            $(".output").text(
              index === -1 ? "" : "Image " + paths[index]
            );
          }
        );

        viewer.addHandler("canvas-click", function (event) {
          if (!event.quick) {
            return;
          }

          var index = hitTest(viewer.viewport.pointFromPixel(event.position));
          if (index !== -1) {
            bounds = viewer.world.getItemAt(index).getBounds();
            expansion = Math.floor(bounds.width / 8);
            bounds.width = bounds.width + expansion;
            bounds.height = bounds.height + expansion;
            bounds.x = Math.max(0, bounds.x - Math.floor(expansion / 2));
            bounds.y = Math.max(0, bounds.y - Math.floor(expansion / 2));
            viewer.viewport.fitBounds(bounds);
          }
        });
      }
    </script>
  </body>
</html>''')
        if out_remote:
            fs.put(filesave + exp_id + "_all_viewer.html", dst + exp_id + "_all_viewer.html")
            os.remove(filesave + exp_id + "_all_viewer.html")

    t1 = time.time()
    filepath = "time_stacking.txt"
    with open(filepath, 'a') as f:
        f.write('\n' + str(t1-t0))

def imread_gcsfs(fs,file_path):
    '''    
    imread_gcsfs gets the image bytes from the remote filesystem and convets it into an image
    
    Arguments:
        fs:         a GCSFS filesystem object
        file_path:  a string containing the GCSFS path to the image (e.g. 'gs://data/folder/image.bmp')
    Returns:
        I:          an image object
    
    This code has no side effects    
    '''
    img_bytes = fs.cat(file_path)
    im_type = file_path.split('.')[-1]
    I = imageio.core.asarray(imageio.v2.imread(img_bytes, im_type))
    return I

def generate_dpc(I1,I2,use_gpu=False):
	if use_gpu:
		# img_dpc = cp.divide(img_left_gpu - img_right_gpu, img_left_gpu + img_right_gpu)
		# to add
		I_dpc = 0
	else:
		I_dpc = np.divide(I1-I2,I1+I2)
		I_dpc = I_dpc + 0.5
	I_dpc[I_dpc<0] = 0
	I_dpc[I_dpc>1] = 1
	return I_dpc

def export_spot_images_from_fov(I_fluorescence,I_dpc,spot_data,parameters,settings,gcs_settings,dir_out=None,r=30,generate_separate_images=False):
	pass
	# make I_dpc RGB
	if(len(I_dpc.shape)==3):
		# I_dpc_RGB = I_dpc
		I_dpc = I_dpc[:,:,1]
	else:
		# I_dpc_RGB = np.dstack((I_dpc,I_dpc,I_dpc))
		pass
	# get overlay
	# I_overlay = 0.64*I_fluorescence + 0.36*I_dpc_RGB
	# get the full image size
	height,width,channels = I_fluorescence.shape
	# go through spot
	counter = 0
	
	for idx, entry in spot_data.iterrows():
		# get coordinate
		i = int(entry['FOV_row'])
		j = int(entry['FOV_col'])
		x = int(entry['x'])
		y = int(entry['y'])
		# create the arrays for cropped images
		I_DPC_cropped = np.zeros((2*r+1,2*r+1), np.float)
		I_fluorescence_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		# I_overlay_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		# identify cropping region in the full FOV 
		x_start = max(0,x-r)
		x_end = min(x+r,width-1)
		y_start = max(0,y-r)
		y_end = min(y+r,height-1)
		x_idx_FOV = slice(x_start,x_end+1)
		y_idx_FOV = slice(y_start,y_end+1)
		# identify cropping region in the cropped images
		x_cropped_start = x_start - (x-r)
		x_cropped_end = (2*r+1-1) - ((x+r)-x_end)
		y_cropped_start = y_start - (y-r)
		y_cropped_end = (2*r+1-1) - ((y+r)-y_end)
		x_idx_cropped = slice(x_cropped_start,x_cropped_end+1)
		y_idx_cropped = slice(y_cropped_start,y_cropped_end+1)
		# do the cropping 
		I_DPC_cropped[y_idx_cropped,x_idx_cropped] = I_dpc[y_idx_FOV,x_idx_FOV]
		I_fluorescence_cropped[y_idx_cropped,x_idx_cropped,:] = I_fluorescence[y_idx_FOV,x_idx_FOV,:]
		
		# combine
		if counter == 0:
			I = np.dstack((I_fluorescence_cropped,I_DPC_cropped))[np.newaxis,:]
			if generate_separate_images:
				I_DAPI = I_fluorescence_cropped[np.newaxis,:]
				I_DPC = I_DPC_cropped[np.newaxis,:]
		else:
			I = np.concatenate((I,np.dstack((I_fluorescence_cropped,I_DPC_cropped))[np.newaxis,:]))
			if generate_separate_images:
				I_DAPI = np.concatenate((I_DAPI,I_fluorescence_cropped[np.newaxis,:]))
				I_DPC = np.concatenate((I_DPC,I_DPC_cropped[np.newaxis,:]))
		counter = counter + 1

	if counter == 0:
		print('no spot in this FOV')
	else:
		# gcs
		if settings['save to gcs']:
			fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
			dir_out = settings['bucket_destination'] + '/' + settings['dataset_id'] + '/' + 'spot_images_fov'

		# convert to xarray
		# data = xr.DataArray(I,coords={'c':['B','G','R','DPC']},dims=['t','y','x','c'])
		data = xr.DataArray(I,dims=['t','y','x','c'])
		data = data.expand_dims('z')
		data = data.transpose('t','c','z','y','x')
		data = (data*255).astype('uint8')
		ds = xr.Dataset({'spot_images':data})
		# ds.spot_images.data = (ds.spot_images.data*255).astype('uint8')
		if settings['save to gcs']:
			store = fs.get_mapper(dir_out + '/' + str(i) + '_' + str(j) + '.zarr')
		else:
			store = dir_out + '/' + str(i) + '_' + str(j) + '.zarr'
		ds.to_zarr(store, mode='w')

		if generate_separate_images:
			
			data = xr.DataArray(I_DAPI,dims=['t','y','x','c'])
			data = data.expand_dims('z')
			data = data.transpose('t','c','z','y','x')
			data = (data*255).astype('uint8')
			ds = xr.Dataset({'spot_images':data})
			if settings['save to gcs']:
				store = fs.get_mapper(dir_out + '/' + str(i) + '_' + str(j) + '_fluorescence.zarr')
			else:
				store = dir_out + '/' + str(i) + '_' + str(j) + '_fluorescence.zarr'
			ds.to_zarr(store, mode='w')

			data = xr.DataArray(I_DPC,dims=['t','y','x'])
			data = data.expand_dims(('z','c'))
			data = data.transpose('t','c','z','y','x')
			data = (data*255).astype('uint8')
			ds = xr.Dataset({'spot_images':data})
			if settings['save to gcs']:
				store = fs.get_mapper(dir_out + '/' + str(i) + '_' + str(j) + '_DPC.zarr')
			else:
				store = dir_out + '/' + str(i) + '_' + str(j) + '_DPC.zarr'
			ds.to_zarr(store, mode='w')

if __name__ == '__main__':
    main()