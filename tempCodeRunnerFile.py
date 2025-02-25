
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
            print("RGB Frame