// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include "benchmark.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov3(const char * paramFile, const char * binFile, const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov3;

    yolov3.opt.use_vulkan_compute = true;

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    // yolov3.load_param("mobilenetv2_yolov3.param");
    // yolov3.load_model("mobilenetv2_yolov3.bin");
    yolov3.load_param(paramFile);
    yolov3.load_model(binFile);
    printf("Model loaded.\n");

    const int target_size = 416;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    /*const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};*/
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    double start = ncnn::get_current_time();
    for (int c = 0; c < 20; ++c) {

        ncnn::Extractor ex = yolov3.create_extractor();

        int r = ex.input("data", in);

        ncnn::Mat out;
        ex.extract("output", out);

        printf("out w/h/c: %d %d %d\n", out.w, out.h, out.c);
        objects.clear();
        for (int i = 0; i < out.h; i++)
        {
            const float* values = out.row(i);

            Object object;
            object.label = values[0];
            object.prob = values[1];
            object.rect.x = values[2] * img_w;
            object.rect.y = values[3] * img_h;
            object.rect.width = values[4] * img_w - object.rect.x;
            object.rect.height = values[5] * img_h - object.rect.y;
            printf("label: %f, prob: %f\n", values[0], values[1]);

            objects.push_back(object);
        }
    }
    double end = ncnn::get_current_time();
    double time = end - start;
    printf("Average inference time:%7.2f \n", time/20);

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	/*
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };
	*/
    static const char* class_names[] = {"person",
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
										"toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label-1], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath] [param file path] [bin file path]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov3(argv[2], argv[3], m, objects);

    draw_objects(m, objects);

    return 0;
}
