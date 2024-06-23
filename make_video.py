#!/usr/bin/env python3
import sys
import torch
import numpy as np
import pickle
import tempfile
import subprocess as sp
from tqdm import tqdm
import cv2
from sklearn.decomposition import PCA, non_negative_factorization
import argparse

# 32, 32, m, n

def color2txt (c):
    return '#%02X%02X%02X' % (c[0], c[1], c[2])

def extract_level_colors (steps, level):
    samples = []
    for i, step in enumerate(steps):
        samples.append(np.reshape(step[level], (32, -1)))
    offset = samples[0].shape[1]
    samples = np.concatenate(samples, axis=1).T

    if False:
        pca = PCA(3)
        samples = pca.fit_transform(samples)
        samples -= np.mean(samples, axis=1, keepdims=True)
        samples /= np.std(samples, axis=1, keepdims=True)
        samples *= 128 #/3.0
        samples += 128
        samples = np.clip(np.rint(samples), 0,255).astype(np.uint8)
    else:
        assert samples.shape[1] == 32
        r = np.linalg.norm(samples[:, 0:11], axis=1)
        g = np.linalg.norm(samples[:, 11:22], axis=1)
        b = np.linalg.norm(samples[:, 22:32], axis=1)
        samples = np.stack([r,g,b], axis=1)
        m2 = np.sqrt(np.mean(np.square(samples), axis=0, keepdims=True))
        samples /= m2
        samples *= 512
        print(np.amax(samples, axis=0))
        print(np.percentile(samples, 95, axis=0))
        samples = np.clip(np.rint(samples),1,255).astype(np.uint8)

    return samples

def render_image (tokens, colors = None):
    ansi = tokens
    if not colors is None:
        assert len(colors.shape) == 2
        assert colors.shape[1] == 3
        ansi = []
        for i in range(colors.shape[0]):
            r,g,b = colors[i, :]
            ansi.append("\033[1;%d;%d;%d;t%s" % (r,g,b,str(tokens[i])))
        ansi.append("\033[0m")
        ansi.append(tokens[colors.shape[0]])
        ansi.append("\033[1;1;1;1;t")

        for i in range(colors.shape[0]+1, len(tokens)):
            ansi.append(tokens[i]) #' ' * len(tokens[i]))

    text = ''.join(ansi)
    with tempfile.NamedTemporaryFile(mode='w') as f, \
         tempfile.NamedTemporaryFile() as f2:
        f.write(text)
        f.flush()
        sp.check_call('ansilove -o %s %s > /dev/null 2> /dev/null' % (f2.name, f.name), shell=True)
        return cv2.imread(f2.name, cv2.IMREAD_COLOR)

def process_level (tokens, steps, level, image_shape):
    colors = extract_level_colors(steps, level)
    off = 0
    images = []
    for i, step in tqdm(list(enumerate(steps))):
        _, a, b, c = step.shape

        step_colors = np.reshape(colors[off:(off + b * c), :], (b, c, colors.shape[-1]))
        off += b * c
        if b > 1: # skip prompt
            continue
        image = render_image(tokens, step_colors[0])
        if image.shape != image.shape:
            cv2.imwrite('b.png', image)
            sys.exit(0)

        images.append(image)
    return images

def montage_level (images, t, h, w):
    #R = 5
    #C = 6
    R = 1
    C = 1
    assert R * C == len(images)
    #h, w, _ = images[0][0].shape
    H = h * R
    W = w * C
    out = np.zeros((H, W, 3), dtype=np.uint8)


    l = 0
    for r in range(R):
        for c in range(C):
            if l < len(images):
                image = images[l][t]
                h1, w1, _ = image.shape
                out[(r*h):(r*h+h1), (c*w):(c*w+w1), :] = image
            l += 1
    return out




def make_video (input_path, output_path, level):
    with open(input_path, 'rb') as f:
        tokens, steps = pickle.load(f)
    # steps[0].shape        levels * channels * n * n
    #   n == number of prompt tokens
    levels, channels, prompt_length, _ = steps[0].shape

    frame = render_image(tokens, None)
    frames_by_level = []
    #for level in range(levels):
    while True:
        print("Level:", level)
        frames_by_level.append(process_level(tokens, steps, level, frame.shape))
        break

    T = len(frames_by_level[0])
    out = None
    for t in tqdm(list(range(T))):
        #frame = montage_level(frames_by_level, t)
        frame = frames_by_level[0][t]
        H, W, _ = frame.shape
        if out is None:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 2.0, (W, H))
        out.write(frame)
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='output.pkl', type=str)
    parser.add_argument("-o", "--output", default='output.avi', type=str)
    parser.add_argument("-l", "--level", default=15, type=int)
    args = parser.parse_args()
    make_video(args.input, args.output, args.level)

