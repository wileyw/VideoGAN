






import cv2
import glob
import os

INPUT_DIR = '/home/michael/repo1/2011_09_26/2011_09_26_drive_0048_sync/image_00/data'
OUTPUT_DIR = 'output'

def draw_flow(image, flow):
    size_u, size_v, channels = flow.shape

    for u in range(0, size_u, 5):
        for v in range(0, size_v, 5):
            du = flow[u, v, 0]
            dv = flow[u, v, 1]
            cv2.line(image, (v, u), (int(v + dv), int(u + du)), (0, 255, 0))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_folders = ['/home/michael/repo1/2011_09_26/2011_09_26_drive_0048_sync/image_00/data']
    for video_folder in video_folders:
        image_paths = sorted(glob.glob(os.path.join(video_folder, '*.png')))
        for i in range(0, len(image_paths) - 1, 1):
            prev_image_path = image_paths[i]
            next_image_path = image_paths[i + 1]

            prev_image_color = cv2.imread(prev_image_path)
            next_image_color = cv2.imread(next_image_path)

            prev_image = cv2.cvtColor(prev_image_color, cv2.COLOR_RGB2GRAY)
            next_image = cv2.cvtColor(next_image_color, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=2, poly_n=7, poly_sigma=1.5, flags=0)

            draw_flow(prev_image_color, flow)

            cv2.imshow('image', next_image)
            cv2.imshow('flow', prev_image_color)
            cv2.waitKey(0)

        break

    print('test')

if __name__ == '__main__':
    main()

