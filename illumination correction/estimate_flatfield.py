import pybasic
from matplotlib import pyplot as plt
import numpy as np

# run in the folder of the single channel images

images = pybasic.tools.load_data('./','.bmp',verbosity=True)
flatfield, darkfield = pybasic.basic(images, darkfield=False)

np.save('flatfield.npy',flatfield)
np.save('darkfield.npy',darkfield)

plt.title('Flatfield')
plt.imshow(flatfield)
plt.colorbar()
plt.savefig('flatfield.png')

plt.title('Darkfield')
plt.imshow(darkfield)
plt.colorbar()
plt.savefig('darkfield.png')

'''
# correction
import glob
import cv2
flatfield = np.load('flatfield.npy')
files = glob.glob('.' + '/' + '*.bmp')
if files:
	print(files[0])
for file in files:
	I = cv2.imread(file)
	I = I[:,:,0]
	I = I.astype('float')
	I = I/flatfield
	cv2.imwrite(file.replace('.bmp','')+'_corrected.bmp',I.astype('uint8'))
	print('----------')
	print(file)
'''