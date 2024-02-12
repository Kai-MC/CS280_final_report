import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import re
from PIL import Image



DEVICE = 'cuda'


PERCENTILE_THRESHOLD = 60  # Set the desired percentile here, for example, 50 for the 50th percentile

def load_image(imfile):
    img = Image.open(imfile).convert('RGB')  # Convert image to RGB
    img = np.array(img).astype(np.float32)  # Convert to float for further processing

    # Convert to grayscale for gradient computation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Compute the gradients using a Sobel filter
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    G = np.sqrt(sobel_y**2 + sobel_x**2) * 2

    # Find the percentile value
    percentile_value = np.percentile(G, PERCENTILE_THRESHOLD)

    # Normalize and convert back to uint8
    sobel_y = cv2.convertScaleAbs(G)

    # Apply percentile threshold: Set gradient values below the threshold to zero
    sobel_y[sobel_y < percentile_value] = 0

    # Replace the blue channel with the y-gradient
    img[..., 2] = sobel_y  # Replace the blue channel (index 2 in RGB format)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image_origin(imfile):
    img = Image.open(imfile).convert('RGB')  # Convert image to RGB
    img = np.array(img).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)




def viz(img, origin, flo, lower_percentile=25, upper_percentile=75, frame_index = 0):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    origin = origin[0].permute(1, 2, 0).cpu().numpy()
    
    # Calculate the magnitude of flow vectors
    magnitude = np.sqrt(flo[..., 0]**2 + flo[..., 1]**2)

    # Calculate lower and upper magnitude thresholds based on percentiles
    lower_magnitude_threshold = np.percentile(magnitude, lower_percentile)
    upper_magnitude_threshold = np.percentile(magnitude, upper_percentile)

    # Create a mask for flow vectors where the magnitude is within the specified range
    mask = (magnitude > lower_magnitude_threshold) & (magnitude < upper_magnitude_threshold)
    
    # Apply the mask to the flow vectors
    thresholded_flo = flo * mask[..., np.newaxis] 
    
    # Map thresholded flow to rgb image
    thresholded_flo_rgb = flow_viz.flow_to_image(thresholded_flo)
    #img_flo = np.concatenate([img, thresholded_flo_rgb], axis=0) # display the gradient image
    img_flo = np.concatenate([origin, thresholded_flo_rgb], axis=0) # display the origin image

    # Convert to grayscale for display
    img_flo_gray = cv2.cvtColor(img_flo.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Display the grayscale image
    #cv2.imshow('image', img_flo_gray)
    #cv2.waitKey()
    # Check if the output folder exists, if not, create it
    output_folder = 'output_image_2579_color'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image
    output_path = os.path.join(output_folder, f'frame_{frame_index}.png')
    #cv2.imwrite(output_path, img_flo_gray)
    cv2.imwrite(output_path, img_flo)



def extract_number(filename):
    # This function extracts the number from filenames like 'frame_0', 'frame_1', etc.
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else None

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images, key=extract_number)
        index = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            #image1 = load_image(images[0])
            image2 = load_image(imfile2)
            origin_image = load_image_origin(imfile1)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, origin_image, flow_up, 80,100, index)
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)


