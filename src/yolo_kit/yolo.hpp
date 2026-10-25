#ifndef YOLO_API_H
#define YOLO_API_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>

#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

#ifdef __cplusplus
extern "C"{
#endif

struct YoloDetection
{
    int32_t class_id;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};

struct YoloConfig
{
    std::string model_path;
    int class_num;
    float score_threshold;
    float nms_threshold;
    // Explicit decoding contract from the model exporter; no guessed defaults.
    std::string score_format; // "logits" or "probability"
    std::string box_format;   // "ltrb_grid", "ltrb_pixels", "xyxy_pixels", "cxcywh_pixels"

};

struct YoloContext
{
    YoloConfig config;
};

int yolo_init(const char* config_path);
// data: tightly packed uint8 NV12, Y followed by interleaved UV.
// width/height: original image dimensions, both positive and even.
// data_size: at least width * height * 3 / 2 bytes; padded rows are unsupported.
// Calls must be serialized. data must remain valid until this call returns.
// out_count is required; results may be null only when capacity is zero.
// On success, results[0..*out_count) holds original-image pixel coordinates.
// Return: 0 on success (including no detections), -1 on error; out_count=0 on error.
// At most capacity boxes are returned, ordered by descending confidence.
int yolo_process_nv12(const uint8_t* data, size_t data_size, 
    int width, int height, 
    YoloDetection* results, 
    int capacity, int* out_count
);

int yolo_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
