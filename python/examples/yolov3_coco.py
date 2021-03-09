import numpy as np
import cv2
import time
import os
import argparse
import ncnn

class_names = ["person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"]

image = cv2.imread(r'C:\Users\DY\Cache\cv_test_video\WIN_6.mp4_20210307_162930.998.png', 1)
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image_resized = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
data = ncnn.Mat.from_pixels_resize(image.data, ncnn.Mat.PixelType.PIXEL_BGR, 1280, 720, 416, 416)
data.substract_mean_normalize([0.0,0.0,0.0],[1/255.0,1/255.0,1/255.0])

net = ncnn.Net()
net.opt.use_vulkan_compute = True
ret = net.load_param(r'C:\Users\DY\Cache\ncnn\yolov3.param')
if ret != 0:
    print('Failed to load param file')

ret = net.load_model(r'C:\Users\DY\Cache\ncnn\yolov3.bin')
if ret != 0:
    print('Failed to load bin file.')

ex = net.create_extractor()

ret = ex.input('data', data)
if ret != 0:
    print('Failed to run input.')

ret, out = ex.extract('output')

for i in range(0, out.h*6, 6):
    print(f'class: {class_names[int(out[i])-1]}, prob: {out[i+1]}')
    
def parser():
    parser = argparse.ArgumentParser(description="YOLOv3 Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--bin_file", default="yolov3.param",
                        help="yolov3 .bin file path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--param_file", default="yolov3.param",
                        help="path to param file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.param_file):
        raise(ValueError("Invalid param path {}".format(os.path.abspath(args.param_file))))
    if not os.path.exists(args.bin_file):
        raise(ValueError("Invalid bin file path {}".format(os.path.abspath(args.bin_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))

def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

'''
def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)
'''

net_width = 416
net_height = 416

def image_detection(image_path, network, ex, class_names, class_colors, thresh):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (net_width, net_height),
                               interpolation=cv2.INTER_LINEAR)

    ex.input("data", ncnn.Mat(image_resized))
    ret, out = ex.extract("output")
    if ret != 0:
        print('Failed to detect objects.')
        return None, None

    '''
    for i in range(out.h):
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;
        printf("label: %f, prob: %f\n", values[0], values[1]);

    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return image_resized, detections
    '''

'''
def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions
'''
'''
def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])
'''
'''
def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))
'''
'''
def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)
'''

def main():
    args = parser()
    check_arguments_errors(args)

    net = ncnn.Net()
    ret = net.load_param(args.param_file)
    if ret != 0:
        print(f'Failed to load param file {args.param_file}.')
        return

    ret = net.load_model(args.bin_file)
    if ret != 0:
        print(f'Failed to load bin file {args.bin_file}.')
        return

    ex = net.create_extractor()

    images = load_images(args.input)

    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_path = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_path, net, ex, class_names, class_colors, args.thresh
            )
        #if args.save_labels:
        #    save_annotations(image_name, image, detections, class_names)
        #print_detections(detections, args.ext_output)
        now = time.time()
        print(f"Inference time: {now-prev_time}s")
        if not args.dont_show:
            cv2.imshow('Result', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        index += 1

if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    #main()
    pass
