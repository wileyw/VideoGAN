import argparse

import cv2
import torch
import numpy as np

import config
import data_util
from g_net import VideoGANGenerator


MODEL_FILEPATH = 'generator_net.pth'
NUM_RECURSIONS = 64
HIST_LEN = 4
CROP_HEIGHT = 32
CROP_WIDTH = 32
DTYPE = config.dtype


def crop_batch(images, fw, fh, cw, ch):
    images = images[0]
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


def reconstruct_frame(image_batch, out_w, out_h):
    generated_images = image_batch

    num_images, channels, cell_h, cell_w = generated_images.shape
    nrows = int(np.ceil(out_w/cell_w))
    ncols = int(np.ceil(out_h/cell_h))
    result = np.zeros((cell_w * nrows, cell_h * ncols, channels), dtype=generated_images.dtype)
    for i in range(nrows):
        for j in range(ncols):
            img_patch = generated_images[i * ncols + j].transpose(1, 2, 0)
            result[i * cell_w: (i + 1) * cell_w,
                   j * cell_h: (j + 1) * cell_h,
                   :] = img_patch

    return result[:out_w, :out_h, :]


def save_to_video(frames, video_filename):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(video_filename, fourcc,
                            20.0, (frames[0].shape[1], frames.shape[0]))
    for frame in frames:
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()


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
    frames = data_util.get_full_clips(args.input_dir, HIST_LEN-1, 1)
    frames = frames[:, 0:200, :, :];
    print("frames.shape is", frames.shape)
    frame_w, frame_h = frames[0].shape[0:2]

    # Set initial frames.
    input_frames = data_util.denormalize_frames(frames)
    # input_batched = crop_batch(frames, frame_w, frame_h,
    #                            CROP_WIDTH, CROP_HEIGHT)
    input_batched = input_frames
    # input_batched = np.array([input_frames[0, :, :, :3], input_frames[0, :, :, 3:6],
    #                           input_frames[0, :, :, 6:9], input_frames[0, :, :, 9:]])

    # Initialize output frames.
    output_frames = [input_frames[0, :, :, :3], input_frames[0, :, :, 3:6],
                     input_frames[0, :, :, 6:9], input_frames[0, :, :, 9:]]

    # Run inference for length of recursions.
    for i in range(NUM_RECURSIONS):
        print('{} of {} frames'.format(i + 1, NUM_RECURSIONS))

        # Run inference and reconstruct frame.
        input_batched_tensor = torch.tensor(np.rollaxis(input_batched, 3, 1)).type(DTYPE)
        result = generator(input_batched_tensor).detach().numpy()

        # Post-process frame
        result_reconst = reconstruct_frame(result, frame_w, frame_h)
        result_denorm = data_util.denormalize_frames(result_reconst)
        output_frames.append(result_denorm)

        # Setup next batch.
        input_batched = np.concatenate((input_batched, result.transpose(0, 2, 3, 1)), axis=3)
        input_batched = input_batched[:, :, :, 3:]

    # Save frames.
    save_to_video(output_frames, args.video_filename)


if __name__ == '__main__':
    main()
