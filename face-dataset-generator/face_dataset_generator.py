import glob
import os
from typing import Tuple
import logging
from multiprocessing import Pool, Queue

import json
import fire
import cv2
import numpy as np

import pcn
from pcn import utils

logging.basicConfig(level=logging.DEBUG)


class FaceDatasetGeneratorPCN:
    def __init__(
        self,
        output_path: str = '',
        same_size: bool = False,
        crop_size: int = 300,
        padding: int = 20,
        write_json: bool = False,
        write_image: bool = False,
        line_thickness: int = 2,
        side_color: Tuple = (255, 255, 0),
        top_color: Tuple = (255, 0, 0),
    ) -> None:
        super(FaceDatasetGeneratorPCN, self).__init__()
        self.output_path = output_path
        self.same_size = same_size
        self.crop_size = crop_size
        self.padding = padding
        self.write_json = write_json
        self.write_image = write_image
        self.line_thickness = line_thickness
        self.side_color = side_color
        self.top_color = top_color

    def single_image_process(
        self,
        input_path: str,
        show_image: bool = True,
        show_cropped: bool = False,
        info: bool = False,
        show_delay: int = 500,
    ) -> None:
        try:
            os.makedirs(self.output_path, exist_ok=True)
            file_name, ext = os.path.basename(input_path).split('.')
            print(file_name, ext)

            img = cv2.imread(input_path)
            draw_img = img.copy()
            winlist = pcn.detect(img)

            detection_data_dict = {}

            for idx, bbox in enumerate(winlist):
                x1 = bbox.x
                y1 = bbox.y
                xwh = bbox.width
                xwh_half = xwh // 2

                px1 = x1 - self.padding
                py1 = y1 - self.padding
                px2 = x1 + xwh + self.padding
                py2 = y1 + xwh + self.padding

                corner_points_list = [
                    (px1, py1),
                    (px1, py2),
                    (px2, py2),
                    (px2, py1)
                ]

                center_x = x1 + xwh_half
                center_y = y1 + xwh_half

                rotated_corner_points_list = [
                    utils.rotate_point(x, y, center_x, center_y, bbox.angle) for x, y in corner_points_list
                ]

                if self.write_json:
                    unrotated_dict = {
                        'x1': px1,
                        'y1': py1,
                        'wh': xwh + self.padding,
                        'angle': bbox.angle
                    }
                    rotated_dict = {
                        'p1': rotated_corner_points_list[0],
                        'p2': rotated_corner_points_list[1],
                        'p3': rotated_corner_points_list[2],
                        'p4': rotated_corner_points_list[3],
                    }
                    detection_data_dict[idx] = [unrotated_dict, rotated_dict]

                if show_image:
                    cv2.line(draw_img, rotated_corner_points_list[0], rotated_corner_points_list[1], self.side_color,
                             self.line_thickness)
                    cv2.line(draw_img, rotated_corner_points_list[1], rotated_corner_points_list[2], self.side_color,
                             self.line_thickness)
                    cv2.line(draw_img, rotated_corner_points_list[2], rotated_corner_points_list[3], self.side_color,
                             self.line_thickness)
                    cv2.line(draw_img, rotated_corner_points_list[3], rotated_corner_points_list[0], self.top_color,
                             self.line_thickness)

                if self.same_size:
                    self.crop_size = xwh + self.padding

                src_triangle = np.array(
                    [
                        rotated_corner_points_list[0],
                        rotated_corner_points_list[1],
                        rotated_corner_points_list[2],
                    ],
                    dtype=np.float32
                )
                dst_triangle = np.array(
                    [
                        (0, 0),
                        (0, self.crop_size - 1),
                        (self.crop_size - 1, self.crop_size - 1),
                    ],
                    dtype=np.float32
                )
                rot_mat = cv2.getAffineTransform(src_triangle, dst_triangle)
                ret = cv2.warpAffine(img, rot_mat, (self.crop_size, self.crop_size))

                if show_image and show_cropped:
                    cv2.imshow("Image", ret)
                    cv2.waitKey(show_delay)
                    cv2.destroyAllWindows()

                if self.write_image and self.output_path != '':
                    out_file_name = f'{file_name}_out_{idx}.{ext}'
                    if info:
                        print(os.path.join(self.output_path, out_file_name))
                    cv2.imwrite(os.path.join(self.output_path, out_file_name), ret)

            if self.write_json:
                if info:
                    print(os.path.join(self.output_path, f'{file_name}.json'))
                try:
                    with open(os.path.join(self.output_path, f'{file_name}.json'), 'w') as f:
                        json.dump(detection_data_dict, f, indent=4, sort_keys=True)
                except Exception as e:
                    logging.exception(e)

            if show_image and show_image:
                cv2.imshow("Image", draw_img)
                cv2.waitKey(show_delay)
                cv2.destroyAllWindows()

        except Exception as e:
            logging.exception(e)

    def batch_image_process(
            self,
            input_path: str,
            ext: str = '*.jpg',
    ) -> None:
        assert os.path.isdir(input_path), 'Please enter a directory with images.'

        files_list = glob.glob(os.path.join(input_path, ext))
        for f in files_list:
            self.single_image_process(
                input_path=f,
                info=False,
                show_image=False,
                show_cropped=False,
            )

    def process_queue(self, queue):
        while not queue.empty():
            filename = queue.get()
            self.single_image_process(
                input_path=filename,
                info=False,
                show_image=False,
                show_cropped=False,
            )

    def mp_batch_image_process(
            self,
            input_path: str,
            processes: int = os.cpu_count() - 1,
            ext: str = '*.jpg',
    ) -> None:
        assert os.path.isdir(input_path), 'Please enter a directory with images.'
        assert processes <= os.cpu_count(), 'Please enter less than the number of CPUs.'

        files_list = glob.glob(os.path.join(input_path, ext))

        task_queue = Queue()
        for f in files_list:
            task_queue.put(f)

        pool = Pool(processes, self.process_queue, (task_queue,))
        pool.close()
        pool.join()

    def video_process(
            self,
    ) -> None:
        raise NotImplementedError()

    def batch_video_process(
            self,
    ) -> None:
        raise NotImplementedError()


if __name__ == '__main__':
    fire.Fire(FaceDatasetGeneratorPCN)
