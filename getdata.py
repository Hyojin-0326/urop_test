# #import os
# import pickle
# import numpy as np

# def load_pkl(file_path):
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#     return data

# # data 폴더의 rgb_data.pkl을 읽음
# base_dir = os.path.dirname(os.path.abspath(__file__))
# pkl_path = os.path.join(base_dir,"data", "rgb_data.pkl")
# data = load_pkl(pkl_path)

# # 데이터 구조 확인
# print(type(data[0]))

# if isinstance(data, list) and isinstance(data[0], np.ndarray):
#     image_array = np.array(data[0])  # 첫 번째 이미지 사용
# else:
#     raise ValueError("pkl 데이터 구조가 예상과 다름!")

import pickle
import os
import numpy as np

def load_pickle_data(file_name):
    """
    지정된 파일명을 가진 pickle 데이터를 불러와 반환
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 py 파일 경로
    file_path = os.path.join(current_dir,"data", file_name)

    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_rgb_depth_data():
    """
    RGB 및 Depth 데이터를 불러와 NumPy 배열로 변환하여 반환
    """
    rgb_data = load_pickle_data("rgb_data.pkl")
    depth_data = load_pickle_data("depth_data.pkl")

    rgb_frame = np.array(rgb_data)
    depth_frame = np.array(depth_data)

    return rgb_frame, depth_frame
