#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pickle
import tempfile
import subprocess as sp
from multiprocessing import Pool
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
        #print(np.amax(samples, axis=0))
        #print(np.percentile(samples, 95, axis=0))
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
        frame = cv2.imread(f2.name, cv2.IMREAD_COLOR)
        return frame

def visualize_series (root, frames, image_scale=2.0, video_scale=2.0):
    os.makedirs(root, exist_ok=True)
    T = len(frames)
    out = None
    with open(os.path.join(root, 'index.html'), 'w') as f:
        #f.write(f'<html><body><h1>Level {level}</h1>\n')
        f.write(f'<html><body>\n')
        for t in range(T):
            frame0 = frames[t]
            frame = frame0
            if not video_scale is None:
                frame = cv2.resize(frame, None, fx=video_scale, fy=video_scale)
            H, W, _ = frame.shape
            if out is None:
                out = cv2.VideoWriter(os.path.join(root, 'video.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 1.0, (W, H))
                f.write(f'''
                <video width="{frame.shape[1]}" height="{frame.shape[0]}" controls>
                <source src="video.mp4" type="video/mp4"/>
                </video><br/>
                ''')
            f.write(f'''<h3>Step {t}</h3>\n''')
            f.write(f'''<img src='{t:04d}.jpg'></img><br/>\n''')
            out.write(frame)
            frame = frame0
            if not image_scale is None:
                frame = cv2.resize(frame, None, fx=image_scale, fy=image_scale)
            cv2.imwrite(os.path.join(root, '%04d.jpg' % t), frame)
    out.release()

def process_level (args):
    tokens, steps, level, output_path = args
    colors = extract_level_colors(steps, level)
    off = 0
    images = []
    for i, step in enumerate(steps):
        _, a, b, c = step.shape

        step_colors = np.reshape(colors[off:(off + b * c), :], (b, c, colors.shape[-1]))
        off += b * c
        if b > 1: # skip prompt
            continue
        image = render_image(tokens, step_colors[0])
        images.append(image)

    root = os.path.join(output_path, 'level_%02d' % level)
    visualize_series(root, images)
    return level, images

def visualize (input_path, output_path, level):
    with open(input_path, 'rb') as f:
        tokens, steps = pickle.load(f)
    # steps[0].shape        levels * channels * n * n
    #   n == number of prompt tokens
    levels, channels, prompt_length, _ = steps[0].shape

    frames_by_level = [None for _ in range(levels)]

    os.makedirs(output_path, exist_ok=True)

    with Pool() as pool, \
        open(os.path.join(output_path, 'index.html'), 'w') as f:

        tasks = [(tokens, steps, level, output_path) for level in range(levels)]
        f.write("<html><body>\n")
        f.write(f"<h2><a href='all/'>All Level View</a></h2>\n")
        l = 0
        for i, frames in tqdm(pool.imap_unordered(process_level, tasks), total=levels):
            frames_by_level[i] = frames
            f.write(f"<h2><a href='level_{l:02d}/'>Level {l}</a></h2>\n")
            l += 1
            pass
        f.write("</body></html>\n")

    series = []
    R, C = 6, 5
    assert R * C == levels
    h, w, _ = frames_by_level[0][0].shape
    H = h * R
    W = w * C
    for t, _ in enumerate(frames_by_level[0]):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        l = 0
        for r in range(R):
            for c in range(C):
                if l < len(frames_by_level):
                    image = frames_by_level[l][t]
                    h1, w1, _ = image.shape
                    frame[(r*h):(r*h+h1), (c*w):(c*w+w1), :] = image
                l += 1
        series.append(frame) #cv2.resize(frame, None, fx=0.4, fy=0.4))
    visualize_series(os.path.join(output_path, 'all'), series, image_scale=None, video_scale=0.4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='output.pkl', type=str)
    parser.add_argument("-o", "--output", default='output', type=str)
    parser.add_argument("-l", "--level", default=15, type=int)
    args = parser.parse_args()
    visualize(args.input, args.output, args.level)
    print("Run the following command to update video:")
    print("ls -d level_* | parallel ffmpeg -i {}/video.avi -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p {}/video.mp4")

