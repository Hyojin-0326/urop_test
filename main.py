import numpy as np
import cv2
import pyrealsense2 as rs
import getdata
import utils

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    classes = open(names_path).read().strip().split("\n")
    
    try:
        while True:
            rgb_frame, depth_frame = getdata.get_rgb_depth_data()
            print("RGB Frame Shape:", rgb_frame.shape)
            print("Depth Frame Shape:", depth_frame.shape)

            if depth_frame is None or rgb_frame is None:
                print("프레임이 없습니다.")
            
            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            color_image = np.asanyarray(rgb_frame)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_image *= depth_scale  

            filled_depth = utils.bilinear_interpolation_kdtree(depth_image)
            
            H, W = color_image.shape[:2]
            blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            boxes, confidences, class_ids = [], [], []
            conf_threshold = 0.5
            nms_threshold = 0.4
            for output in outs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * W)
                        center_y = int(detection[1] * H)
                        w = int(detection[2] * W)
                        h = int(detection[3] * H)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            for i in indices:
                i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
                x, y, w, h = boxes[i]
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(x+w, W)
                y_max = min(y+h, H)
                
                top_edge = filled_depth[y_min, x_min:x_max]
                bottom_edge = filled_depth[y_max-1, x_min:x_max]
                top_edge = top_edge[top_edge > 0]
                bottom_edge = bottom_edge[bottom_edge > 0]
                if len(top_edge) == 0 or len(bottom_edge) == 0:
                    continue
                avg_top_depth = np.mean(top_edge)
                avg_bottom_depth = np.mean(bottom_edge)
                
                _, top_y, _ = utils.pixel_to_3d((x_min + x_max) // 2, y_min, avg_top_depth, fx, fy, cx, cy)
                _, bottom_y, _ = utils.pixel_to_3d((x_min + x_max) // 2, y_max - 1, avg_bottom_depth, fx, fy, cx, cy)
                object_height = abs(bottom_y - top_y)
                
                label = f"{classes[class_ids[i]]}: {object_height:.2f} m"
                cv2.putText(color_image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("RGB", color_image)
            if cv2.waitKey(1) == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
