import glob
from PIL import Image
import imageio


def make_gif(frame_folder, fps):
	print("gif")
	
	giffile = 'gif.gif'
	
	images_data = []
	for i in range(43):
		data = imageio.imread(f'gif_folder/frame_{i}.jpg')
		images_data.append(data)
	
	imageio.mimwrite(giffile, images_data, format='.gif', fps=fps)

