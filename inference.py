import argparse

import cv2
import torch
import numpy as np

import config
import data_util
from vanilla_gan.video_gan import VideoGANGenerator


MODEL_FILEPATH = 'generator_net.pth'
NUM_RECURSIONS = 2# 64
HIST_LEN = 4
CROP_HEIGHT = 32
CROP_WIDTH = 32
DTYPE = config.dtype

def crop_batch(images, fw, fh, cw, ch):
    batch = []
    for i in range(int(np.ceil(fw/cw))):
        for j in range(int(np.ceil(fh/ch))):
            img = np.zeros([cw, ch, images.shape[2]])
            x = cw
            if ((i+1) * cw) > fw:
                x = fw - (i * cw)
            y = ch
            if ((j+1) * ch) > fh:
                y = fh - (j * ch)
            img[:x, :y, :] = images[i * cw: min(fw, (i + 1) * cw),
                                    j * ch: min(fh, (j + 1) * ch),
                                    :]
            batch.append(img)
    batch = np.stack(batch, axis=0)
    return batch


def reconstruct_frame(generated_images, fw, fh, ch, cw):
    out = np.empty([fw, fh, 3])

    num_images, channels = generated_images.shape[:2]
    ncols = int(np.ceil(fw/cw))
    nrows = int(np.ceil(fh/ch))
    for i in range(nrows):
        for j in range(ncols):
            x = (i+1)*cw
            x1 = cw
            if x > fw:
                x1 = fw - (i * cw)
                x = fw
            y = (j+1)*ch
            y1 = ch
            if y > fh:
                y1 = fh - (j * ch)
                y = fh
            print(i*cw, x, x1, j*ch, y, y1)
            out[i*cw:x, j*ch:y, :] = generated_images[i*ncols+j].transpose(1, 2, 0)[:x1, :y1, :]
    return out

def save_to_video(frames, video_filename):
    video = cv2.VideoWriter(video_filename, -1,
                            1, frames[0].shape[:2])
    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def main():
    parser = argparse.ArgumentParser(
        description="""
            Script to run inference on a random input seeding
            of videoGAN
        """)
    parser.add_argument(
        'video_filename',
        help='output filename of video')
    parser.add_argument(
        '-i', '--input_dir',
        default='Ms_Pacman/Test',
        help='input dir of training to take random images from')


    args = parser.parse_args()

    # Load generator.
    generator = VideoGANGenerator()
    generator.load_state_dict(torch.load(MODEL_FILEPATH))
    generator.eval()

    # Load input seed.
    frames = data_util.get_full_clips(args.input_dir, HIST_LEN, 1)
    frame_w, frame_h = frames[0].shape

    # Set initial frames.
    input_frames = data_util.denormalize_frames(frames)
    input_batched = crop_batch(frames, frame_w, frame_h, CROP_WIDTH, CROP_HEIGHT)

    # Initialize output frames.
    output_frames = [input_frames[:,:,:3], input_frames[:,:,3:6],
                     input_frames[:,:,6:9], input_frames[:,:,9:]]

    # Run inference for length of recursions.
    for i in range(NUM_RECURSIONS):
        print('{} of {} frames'.format(i + 1, NUM_RECURSIONS))
        print(input_batched.shape)
        # Run inference and reconstruct frame.
        input_batched_tensor = torch.tensor(np.rollaxis(input_batched, 3, 1)).type(DTYPE)
        result = generator(input_batched_tensor).detach().numpy()

        result_reconst = reconstruct_frame(result, frame_w, frame_h, CROP_WIDTH, CROP_HEIGHT)
        print(result_reconst.shape)
        result_denorm = data_util.denormalize_frames(result_reconst)
        output_frames.append(result_denorm)

        # Pop first frame and add result to input.
        input_batched = np.concatenate((input_batched, result), axis=-1)
        input_batched = input_batched[:, :, :, 3:]

    for i, frame in enumerate(output_frames):
        cv2.imwrite('test_{}.png'.format(i), frame)
    save_to_video(output_frames, args.video_filename)

if __name__ == '__main__':
    main()
