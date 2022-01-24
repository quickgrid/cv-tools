"""
Command:

"""
import os
import subprocess
import time
from typing import Tuple


input_path: str = 'face-dataset-generator/imgs'
output_path: str = r'face-dataset-generator/outputs'
same_size: bool = False
crop_size: int = 300
padding: int = 20
show_image: bool = True
show_cropped: bool = True
write_json: bool = True
write_image: bool = True
info: bool = True
show_delay: int = 500
line_thickness: int = 2
side_color: Tuple = (255, 255, 0)
top_color: Tuple = (255, 0, 0)

processes: int = os.cpu_count() - 1

start_time = time.time()

subprocess.run(
    (
        'python',
        'face-dataset-generator/face_dataset_generator.py',
        'mp_batch_image_process',
        # 'batch_image_process',
        f'--input_path={input_path}',
        f'--output_path={output_path}',
        f'--write_image={write_image}',
        f'--write_json={write_json}'
        f'--processes={processes}'
    )
)

print(f'TIME {time.time() - start_time}')

# subprocess.run(
#     (
#         'python',
#         'face-dataset-generator/face_dataset_generator.py',
#         'batch_image_process',
#         f'--input_path={input_path}',
#         f'--info={info}',
#     )
# )
