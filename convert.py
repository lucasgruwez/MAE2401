import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy.signal import argrelmin

j = 0

N = 1400

pbar = tqdm(total=N)

for i in range(N):

	pbar.update(1)

	cmd = f'convert -density 288 mae2401.pdf\[{i}\] questions/image_{i:03d}.jpg'
	os.system(cmd)

	# Open image as np array and resize to 75%
	im = Image.open(f'questions/image_{i:03d}.jpg')
	scale = 0.75
	im = im.resize((int(im.size[0]*scale), int(im.size[1]*scale)))
	im = np.array(im)

	# Delete file
	os.remove(f'questions/image_{i:03d}.jpg')

	w = im.shape[1]
	h = im.shape[0]

	# Find lowest value in top quarter of image
	min_val = np.min( im[0:int(h/4), :] )

	# Skip if empty or title page
	if min_val > 250:
		continue

	# histograms
	countx = np.mean(im, axis=1)
	county = np.mean(im, axis=0)

	# Find lowst y-counts in left and right halves of image
	idx_left = np.argmin( county[0:int(w/2)] )
	idx_right = np.argmin( county[int(w/2):] ) + int(w/2)

	# Find any peaks in x-counts below 75
	peaks = argrelmin(countx, order=10)[0]
	peaks = peaks[countx[peaks] < 75]
	
	if len(peaks) == 4:
		
		# Get question (top box)
		question = im[peaks[0]:peaks[1], idx_left:idx_right]

		# Get answers (bottom box)
		answers = im[peaks[2]:peaks[3], idx_left:idx_right]

		# Crop 5px from each side of question and answers
		question = question[5:-5, 5:-5]
		answers  = answers[5:-5, 5:-5]

		# Save images using PIL
		Image.fromarray(question).save(f'questions/question_{j:03d}.png')
		Image.fromarray(answers).save(f'questions/answers_{j:03d}.png')

		j += 1

	if len(peaks) == 2:

		# Continued solution
		answers = im[peaks[0]:peaks[1], idx_left:idx_right]

		# Crop 5px from each side of answers
		answers  = answers[5:-5, 5:-5]

		# Open previous solution
		prev_answer = Image.open(f'questions/answers_{j-1:03d}.png')

		# Ensure they are the same width
		if prev_answer.size[0] != answers.shape[1]:
			width = min(prev_answer.size[0], answers.shape[1])
			prev_answer = prev_answer.crop((0, 0, width, prev_answer.size[1]))
			answers = answers[:, 0:width]

		# Append to previous solution
		answers = np.concatenate((prev_answer, answers), axis=0)

		# Save image using PIL
		Image.fromarray(answers).save(f'questions/answers_{j-1:03d}.png')

	# Progress bar description
	pbar.set_description(f'Questions: {j}')