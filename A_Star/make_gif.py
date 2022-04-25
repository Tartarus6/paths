import glob
from PIL import Image
import imageio


def make_gif(frame_folder, fps, num_frames):
	print("gif")
	
	giffile = 'gif.gif'
	
	images_data = []
	for i in range(num_frames):
		data = imageio.imread(f'gif_folder/frame_{i}.jpg')
		images_data.append(data)
	
	imageio.mimwrite(giffile, images_data, format='.gif', fps=fps)

