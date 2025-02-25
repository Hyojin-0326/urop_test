import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import os

# NumPy 버전 문제 해결
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# 파이프라인 생성
pipeline = rs.pipeline()
config = rs.config()

# 해상도 및 프레임 설정
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



# 스트리밍 시작
profile = pipeline.start(config)

# 프레임 저장 리스트
rgb_frames = []
depth_frames = []
MAX_FRAMES = 500  # 최대 저장할 프레임 개수

try:
    while True:
        # 프레임 받기 (이전 프레임 버리기)
        for _ in range(5):  
            frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("⚠️ 프레임을 수신하지 못했습니다.")
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if depth_image is None or color_image is None:
            print("❌ 이미지가 None입니다. 다시 시도하세요.")
            continue

        print(f"✅ RGB 크기: {color_image.shape}, Depth 크기: {depth_image.shape}")

        # Depth 데이터를 8비트로 변환 (디스플레이용)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 메모리 누수 방지 (최대 500프레임까지만 저장)
        if len(rgb_frames) > MAX_FRAMES:
            del rgb_frames[:100]  # 오래된 데이터 삭제
            del depth_frames[:100]

        rgb_frames.append(color_image)
        depth_frames.append(depth_image)

        # 디스플레이
        cv2.imshow('RGB Video', color_image)
        cv2.imshow('Depth Video', depth_colormap)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    # 데이터 저장
    with open('rgb_data.pkl', 'wb') as f:
        pickle.dump(rgb_frames, f)
    with open('depth_data.pkl', 'wb') as f:
        pickle.dump(depth_frames, f)

    pipeline.stop()
    cv2.destroyAllWindows()
