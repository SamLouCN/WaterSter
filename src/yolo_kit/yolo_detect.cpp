#include "yolo.hpp"
#include <limits>

using namespace std;

static bool initialized = false;
static YoloConfig config;
static hbPackedDNNHandle_t packed_handle = nullptr;
static hbDNNHandle_t model_handle = nullptr;
static hbDNNTaskHandle_t task_handle = nullptr;
static bool task_completed = false;
static hbDNNTensor input_tensor{};
static vector<hbDNNTensor> output_tensors;
static int input_width = 0;
static int input_height = 0;
static int output_order[6] = {-1, -1, -1, -1, -1, -1};

static YoloConfig read_config(const char* config_path)
{
    ifstream file(config_path);

    if (!file.is_open())
    {
        throw runtime_error(string("[Error] Failed to open JSON:") + config_path);
    }

    nlohmann::json json;
    file >> json;

    YoloConfig config;
    config.model_path = json.at("model_path").get<string>();
    config.class_num = json.value("class_num", 1);
    config.score_threshold = json.value("score_threshold", 0.25f);
    config.nms_threshold = json.value("nms_threshold", 0.7f);
    config.score_format = json.value("score_format", string());
    config.box_format = json.value("box_format", string());

    return config;
}

static int validate_config(const YoloConfig& config)
{
    try
    {
        if (config.model_path.empty())
        {
            throw runtime_error("[Error] model_path cannot be empty");
        }
        if (!(config.score_threshold > 0.0f && config.score_threshold < 1.0f))
        {
            throw runtime_error("[Error] Invalid score_threshold");
        }
        if (config.class_num <= 0)
        {
            throw runtime_error("[Error] class_num should be at least 1");
        }
        if (!(config.nms_threshold > 0.0f && config.nms_threshold < 1.0f))
        {
            throw runtime_error("[Error] Invalid nms_threshold");
        }
        if (config.score_format != "logits" && config.score_format != "probability")
        {
            throw runtime_error("[Error] Set score_format to logits or probability from the model export contract");
        }
        if (config.box_format != "ltrb_grid" && config.box_format != "ltrb_pixels" &&
            config.box_format != "xyxy_pixels" && config.box_format != "cxcywh_pixels")
        {
            throw runtime_error("[Error] Set box_format to ltrb_grid, ltrb_pixels, xyxy_pixels or cxcywh_pixels from the model export contract");
        }
        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return -1;
    }
    catch (...)
    {
        cerr << "[Error] Unknown error" << '\n';
        return -1;
    }
}

static void preprocess_nv12(
    const uint8_t* data, int width, int height,
    uint8_t* destination, int input_h, int input_w,
    float& x_scale, float& y_scale, int& x_shift, int& y_shift)
{
    int new_w;
    int new_h;

    if (static_cast<int64_t>(input_w) * height <=
        static_cast<int64_t>(input_h) * width)
    {
        new_w = input_w;
        new_h = max(2, static_cast<int>(
            static_cast<int64_t>(height) * input_w / width) & ~1);
    }
    else
    {
        new_h = input_h;
        new_w = max(2, static_cast<int>(
            static_cast<int64_t>(width) * input_h / height) & ~1);
    }

    x_shift = ((input_w - new_w) / 2) & ~1;
    y_shift = ((input_h - new_h) / 2) & ~1;
    x_scale = static_cast<float>(new_w) / width;
    y_scale = static_cast<float>(new_h) / height;

    const size_t source_y_size = static_cast<size_t>(width) * height;
    const size_t target_y_size = static_cast<size_t>(input_w) * input_h;
    cv::Mat source_y(height, width, CV_8UC1,
                     const_cast<uint8_t*>(data));
    cv::Mat source_uv(height / 2, width / 2, CV_8UC2,
                      const_cast<uint8_t*>(data) + source_y_size);
    cv::Mat target_y(input_h, input_w, CV_8UC1, destination);
    cv::Mat target_uv(input_h / 2, input_w / 2, CV_8UC2,
                      destination + target_y_size);
    const cv::Rect y_rect(x_shift, y_shift, new_w, new_h);
    const cv::Rect uv_rect(x_shift / 2, y_shift / 2, new_w / 2, new_h / 2);

    const auto fill_border = [](cv::Mat& plane, const cv::Rect& rect,
                                const cv::Scalar& color)
    {
        if (rect.y > 0)
            plane.rowRange(0, rect.y).setTo(color);
        if (rect.y + rect.height < plane.rows)
            plane.rowRange(rect.y + rect.height, plane.rows).setTo(color);
        cv::Mat middle = plane.rowRange(rect.y, rect.y + rect.height);
        if (rect.x > 0)
            middle.colRange(0, rect.x).setTo(color);
        if (rect.x + rect.width < plane.cols)
            middle.colRange(rect.x + rect.width, plane.cols).setTo(color);
    };

    fill_border(target_y, y_rect, cv::Scalar(125));
    fill_border(target_uv, uv_rect, cv::Scalar(128, 128));

    cv::Mat y_roi = target_y(y_rect);
    cv::Mat uv_roi = target_uv(uv_rect);
    if (new_w == width && new_h == height)
    {
        source_y.copyTo(y_roi);
        source_uv.copyTo(uv_roi);
    }
    else
    {
        cv::resize(source_y, y_roi, y_roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::resize(source_uv, uv_roi, uv_roi.size(), 0, 0, cv::INTER_LINEAR);
    }
}

extern "C" int yolo_init(const char* config_path)
{
    hbPackedDNNHandle_t loaded_handle = nullptr;
    const char* model_file = nullptr;
    int model_count = 0;
    const char** model_names = nullptr;
    int32_t input_count = 0;
    int32_t output_count = 0;
    const int bbox_channels = 4;
    const int strides[3] = {8, 16, 32};
    bool used[6] = {};
    hbSysMem input_memory{};

    if (initialized)
    {
        cerr << "[Error] Model already initialized" << '\n';
        return -1;
    }

    if (packed_handle != nullptr || model_handle != nullptr || task_handle != nullptr ||
        input_tensor.sysMem[0].virAddr != nullptr || !output_tensors.empty())
    {
        cerr << "[Error] Resources remain; call yolo_cleanup() before initializing again\n";
        return -1;
    }

    if (config_path == nullptr || config_path[0] == '\0')
    {
        cerr << "[Error] config_path cannot be empty" << '\n';
        return -1;
    }

    try
    {
        config = read_config(config_path);
    }
    catch (exception& error)
    {
        cerr << error.what() << '\n';
        goto cleanup;
    }

    if (validate_config(config) != 0)
    {
        goto cleanup;
    }

    model_file = config.model_path.c_str();

    if (hbDNNInitializeFromFiles(&loaded_handle, &model_file, 1) != 0)
    {
        cerr << "[Error] Failed to initialize model" << '\n';
        goto cleanup;
    }
    packed_handle = loaded_handle;

    if (hbDNNGetModelNameList(&model_names, &model_count, packed_handle) != 0)
    {
        cerr << "[Error] Failed to get model name list" << '\n';
        goto cleanup;
    }

    if (model_count <= 0 || model_names == nullptr || model_names[0] == nullptr)
    {
        cerr << "[Error] Model package is empty" << '\n';
        goto cleanup;
    }

    if (model_count > 1)
    {
        cout << "[Warning] Models more than one, using the first one" << '\n';
    }

    if (hbDNNGetModelHandle(&model_handle, packed_handle, model_names[0]) != 0)
    {
        cerr << "[Error] Failed to get model handle" << '\n';
        goto cleanup;
    }

    if (hbDNNGetInputCount(&input_count, model_handle) != 0)
    {
        cerr << "[Error] Cannot get input_count" << '\n';
        goto cleanup;
    }

    if (input_count != 1)
    {
        cerr << "[Error] input_count should be 1" << '\n';
        goto cleanup;
    }

    if (hbDNNGetInputTensorProperties(&input_tensor.properties, model_handle, 0) != 0)
    {
        cerr << "[Error] Failed to get input tensor properties" << '\n';
        goto cleanup;
    }

    if (input_tensor.properties.tensorType != HB_DNN_IMG_TYPE_NV12 ||
        input_tensor.properties.validShape.numDimensions != 4 ||
        input_tensor.properties.validShape.dimensionSize[0] != 1)
    {
        std::cerr << "[Error] Expected batch-1 NV12 input\n";
        goto cleanup;
    }

    input_height = input_tensor.properties.validShape.dimensionSize[2];
    input_width = input_tensor.properties.validShape.dimensionSize[3];
    if (input_height <= 0 || input_width <= 0 ||
        input_height % 32 != 0 || input_width % 32 != 0)
    {
        std::cerr << "[Error] Unsupported input dimensions\n";
        goto cleanup;
    }

    if (hbDNNGetOutputCount(&output_count, model_handle) != 0)
    {
        std::cerr << "[Error] Failed to get output count\n";
        goto cleanup;
    }

    if (output_count != 6)
    {
        std::cerr << "[Error] Expected six outputs\n";
        goto cleanup;
    }

    try
    {
        output_tensors.assign(output_count, hbDNNTensor{});
    }
    catch (exception& error)
    {
        cerr << error.what() << '\n';
        goto cleanup;
    }

    for (int i = 0; i < output_count; ++i)
    {
        if (hbDNNGetOutputTensorProperties(
                &output_tensors[i].properties, model_handle, i) != 0)
        {
            std::cerr << "[Error] Failed to get output properties\n";
            goto cleanup;
        }
    }

    for (int i = 0; i < 6; ++i)
    {
        const int stride = strides[i / 2];
        const int expected_h = input_height / stride;
        const int expected_w = input_width / stride;
        const int expected_c = (i % 2 == 0) ? config.class_num : bbox_channels;
        int match = -1;

        for (int j = 0; j < output_count; ++j)
        {
            const auto& shape = output_tensors[j].properties.validShape;
            if (shape.numDimensions == 4 &&
                shape.dimensionSize[0] == 1 &&
                shape.dimensionSize[1] == expected_h &&
                shape.dimensionSize[2] == expected_w &&
                shape.dimensionSize[3] == expected_c)
            {
                if (match != -1)
                {
                    std::cerr << "[Error] Ambiguous output shapes; "
                                 "explicit mapping is required\n";
                    goto cleanup;
                }
                match = j;
            }
        }

        if (match < 0 || used[match])
        {
            std::cerr << "[Error] Output shape mismatch\n";
            goto cleanup;
        }
        output_order[i] = match;
        used[match] = true;
    }

    if (input_tensor.properties.alignedByteSize <= 0)
    {
        std::cerr << "[Error] Invalid input memory size\n";
        goto cleanup;
    }

    if (hbSysAllocCachedMem(&input_memory, input_tensor.properties.alignedByteSize) != 0)
    {
        std::cerr << "[Error] Failed to allocate input memory\n";
        goto cleanup;
    }
    input_tensor.sysMem[0] = input_memory;

    for (int i = 0; i < output_count; ++i)
    {
        const int size = output_tensors[i].properties.alignedByteSize;
        if (size <= 0)
        {
            std::cerr << "[Error] Invalid output memory size\n";
            goto cleanup;
        }

        hbSysMem output_memory{};
        if (hbSysAllocCachedMem(&output_memory, size) != 0)
        {
            std::cerr << "[Error] Failed to allocate output memory\n";
            goto cleanup;
        }
        output_tensors[i].sysMem[0] = output_memory;
    }

    initialized = true;
    return 0;

cleanup:
    if (yolo_cleanup() != 0)
    {
        cerr << "[Error] Cleanup incomplete; call yolo_cleanup() to retry\n";
    }
    return -1;
}

extern "C" int yolo_process_nv12(const uint8_t* data, size_t data_size,
    int width, int height,
    YoloDetection* results,
    int capacity, int* out_count
)
{
    if (out_count == nullptr)
    {
        cerr << "[Error] out_count cannot be null\n";
        return -1;
    }
    *out_count = 0;

    if (!initialized || task_handle != nullptr || input_tensor.sysMem[0].virAddr == nullptr)
    {
        cerr << "[Error] Model is not initialized\n";
        return -1;
    }
    if (data == nullptr || width <= 0 || height <= 0 ||
        width % 2 != 0 || height % 2 != 0)
    {
        cerr << "[Error] Expected NV12 data with positive even dimensions\n";
        return -1;
    }
    if (capacity < 0 || (capacity > 0 && results == nullptr))
    {
        cerr << "[Error] Invalid result buffer or capacity\n";
        return -1;
    }

    const size_t max_size = numeric_limits<size_t>::max();
    if (static_cast<size_t>(width) > max_size / static_cast<size_t>(height))
    {
        cerr << "[Error] NV12 dimensions overflow\n";
        return -1;
    }
    const size_t pixels = static_cast<size_t>(width) * height;
    if (pixels > max_size - pixels / 2 || data_size < pixels + pixels / 2)
    {
        cerr << "[Error] NV12 data_size is too small or overflows\n";
        return -1;
    }

    if (input_tensor.properties.alignedShape.numDimensions != 4 ||
        input_tensor.properties.alignedShape.dimensionSize[2] != input_height ||
        input_tensor.properties.alignedShape.dimensionSize[3] != input_width)
    {
        cerr << "[Error] Spatially padded NV12 input is not supported\n";
        return -1;
    }
    const size_t input_pixels = static_cast<size_t>(input_width) * input_height;
    const size_t input_bytes = input_pixels + input_pixels / 2;
    if (input_tensor.sysMem[0].memSize < input_bytes)
    {
        cerr << "[Error] Model input buffer is too small\n";
        return -1;
    }

    float x_scale, y_scale;
    int x_shift, y_shift;
    try
    {
        preprocess_nv12(
            data, width, height,
            static_cast<uint8_t*>(input_tensor.sysMem[0].virAddr),
            input_height, input_width,
            x_scale, y_scale, x_shift, y_shift);
    }
    catch (const exception& error)
    {
        cerr << "[Error] NV12 preprocessing failed: " << error.what() << '\n';
        return -1;
    }
    catch (...)
    {
        cerr << "[Error] Unknown NV12 preprocessing error\n";
        return -1;
    }

    if (hbSysFlushMem(&input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN) != 0)
    {
        cerr << "[Error] Failed to clean input cache\n";
        return -1;
    }

    if (output_tensors.size() != 6 || config.class_num != 1)
    {
        cerr << "[Error] Expected six outputs and a single-class model\n";
        return -1;
    }

    hbDNNTensor* output = output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    task_completed = false;

    if (hbDNNInfer(&task_handle, &output, &input_tensor,
                   model_handle, &infer_ctrl_param) != 0)
    {
        cerr << "[Error] Failed to start inference\n";
        if (task_handle != nullptr)
            initialized = false;
        return -1;
    }
    if (task_handle == nullptr)
    {
        cerr << "[Error] Inference returned an empty task handle\n";
        initialized = false;
        return -1;
    }
    if (hbDNNWaitTaskDone(task_handle, 0) != 0)
    {
        cerr << "[Error] Failed to wait for inference; call yolo_cleanup()\n";
        initialized = false;
        return -1;
    }
    task_completed = true;
    if (hbDNNReleaseTask(task_handle) != 0)
    {
        cerr << "[Error] Failed to release task; call yolo_cleanup()\n";
        initialized = false;
        return -1;
    }
    task_handle = nullptr;
    task_completed = false;

    // Postprocess is inline: caller-owned results + out_count return the data.
    try
    {
        const bool cls_is_logits = config.score_format == "logits";
        enum class BoxFormat { LtrbGrid, LtrbPixels, XyxyPixels, CxCyWhPixels };
        BoxFormat box_format;
        if (config.box_format == "ltrb_grid")
            box_format = BoxFormat::LtrbGrid;
        else if (config.box_format == "ltrb_pixels")
            box_format = BoxFormat::LtrbPixels;
        else if (config.box_format == "xyxy_pixels")
            box_format = BoxFormat::XyxyPixels;
        else if (config.box_format == "cxcywh_pixels")
            box_format = BoxFormat::CxCyWhPixels;
        else
        {
            cerr << "[Error] Unknown box format\n";
            return -1;
        }

        vector<cv::Rect2d> boxes;
        vector<float> scores;
        const size_t candidate_count =
            static_cast<size_t>(input_width / 8) * (input_height / 8) +
            static_cast<size_t>(input_width / 16) * (input_height / 16) +
            static_cast<size_t>(input_width / 32) * (input_height / 32);
        boxes.reserve(candidate_count);
        scores.reserve(candidate_count);
        const int strides[3] = {8, 16, 32};

        for (int scale_idx = 0; scale_idx < 3; ++scale_idx)
        {
            const int stride = strides[scale_idx];
            const int h = input_height / stride;
            const int w = input_width / stride;
            const int cls_idx = output_order[scale_idx * 2];
            const int box_idx = output_order[scale_idx * 2 + 1];
            if (cls_idx < 0 || cls_idx >= 6 || box_idx < 0 || box_idx >= 6)
            {
                cerr << "[Error] Invalid output mapping\n";
                return -1;
            }

            const int indices[2] = {cls_idx, box_idx};
            const int channels[2] = {1, 4};
            for (int k = 0; k < 2; ++k)
            {
                const hbDNNTensor& tensor = output_tensors[indices[k]];
                const hbDNNTensorProperties& props = tensor.properties;
                const int c = channels[k];
                const size_t expected_bytes =
                    static_cast<size_t>(h) * w * c * sizeof(float);
                if (props.tensorType != HB_DNN_TENSOR_TYPE_F32 ||
                    props.quantiType != NONE ||
                    props.validShape.numDimensions != 4 ||
                    props.validShape.dimensionSize[0] != 1 ||
                    props.validShape.dimensionSize[1] != h ||
                    props.validShape.dimensionSize[2] != w ||
                    props.validShape.dimensionSize[3] != c ||
                    props.stride[3] != static_cast<int>(sizeof(float)) ||
                    props.stride[2] != c * static_cast<int>(sizeof(float)) ||
                    static_cast<size_t>(props.stride[1]) !=
                        static_cast<size_t>(w) * c * sizeof(float) ||
                    tensor.sysMem[0].virAddr == nullptr ||
                    tensor.sysMem[0].memSize < expected_bytes)
                {
                    cerr << "[Error] Unsupported output properties\n";
                    return -1;
                }
                if (hbSysFlushMem(&output_tensors[indices[k]].sysMem[0],
                                   HB_SYS_MEM_CACHE_INVALIDATE) != 0)
                {
                    cerr << "[Error] Failed to invalidate output cache\n";
                    return -1;
                }
            }

            const float* cls_raw = static_cast<const float*>(
                output_tensors[cls_idx].sysMem[0].virAddr);
            const float* box_raw = static_cast<const float*>(
                output_tensors[box_idx].sysMem[0].virAddr);
            for (int gh = 0; gh < h; ++gh)
            {
                for (int gw = 0; gw < w; ++gw)
                {
                    const size_t index = static_cast<size_t>(gh) * w + gw;
                    const float raw_score = cls_raw[index];
                    if (!std::isfinite(raw_score))
                        continue;
                    float score;
                    if (cls_is_logits)
                    {
                        if (raw_score >= 0.0f)
                            score = 1.0f / (1.0f + std::exp(-raw_score));
                        else
                        {
                            const float e = std::exp(raw_score);
                            score = e / (1.0f + e);
                        }
                    }
                    else
                    {
                        if (raw_score < 0.0f || raw_score > 1.0f)
                        {
                            cerr << "[Error] Classification output is outside probability range\n";
                            return -1;
                        }
                        score = raw_score;
                    }
                    if (score < config.score_threshold)
                        continue;
                    const float* box = box_raw + index * 4;
                    if (!std::isfinite(box[0]) || !std::isfinite(box[1]) ||
                        !std::isfinite(box[2]) || !std::isfinite(box[3]))
                        continue;
                    // Use double intermediates to avoid overflow in box arithmetic.
                    double x1, y1, x2, y2;
                    switch (box_format)
                    {
                    case BoxFormat::LtrbGrid:
                        x1 = (gw + 0.5 - box[0]) * stride;
                        y1 = (gh + 0.5 - box[1]) * stride;
                        x2 = (gw + 0.5 + box[2]) * stride;
                        y2 = (gh + 0.5 + box[3]) * stride;
                        break;
                    case BoxFormat::LtrbPixels:
                        x1 = (gw + 0.5) * stride - box[0];
                        y1 = (gh + 0.5) * stride - box[1];
                        x2 = (gw + 0.5) * stride + box[2];
                        y2 = (gh + 0.5) * stride + box[3];
                        break;
                    case BoxFormat::XyxyPixels:
                        x1 = box[0]; y1 = box[1];
                        x2 = box[2]; y2 = box[3];
                        break;
                    case BoxFormat::CxCyWhPixels:
                        x1 = box[0] - static_cast<double>(box[2]) * 0.5;
                        y1 = box[1] - static_cast<double>(box[3]) * 0.5;
                        x2 = box[0] + static_cast<double>(box[2]) * 0.5;
                        y2 = box[1] + static_cast<double>(box[3]) * 0.5;
                        break;
                    default:
                        cerr << "[Error] Unsupported box format\n";
                        return -1;
                    }
                    if (x2 <= x1 || y2 <= y1)
                        continue;
                    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                    scores.push_back(score);
                }
            }
        }

        vector<int> kept_indices;
        if (!boxes.empty())
            cv::dnn::NMSBoxes(boxes, scores, config.score_threshold,
                              config.nms_threshold, kept_indices);
        // Finish all processing before writing caller memory or printing.
        vector<YoloDetection> detections;
        detections.reserve(min(kept_indices.size(), static_cast<size_t>(capacity)));
        for (int idx : kept_indices)
        {
            if (detections.size() >= static_cast<size_t>(capacity))
                break;
            const cv::Rect2d& box = boxes[idx];
            double x1 = (box.x - x_shift) / x_scale;
            double y1 = (box.y - y_shift) / y_scale;
            double x2 = (box.x + box.width - x_shift) / x_scale;
            double y2 = (box.y + box.height - y_shift) / y_scale;
            if (!std::isfinite(x1) || !std::isfinite(y1) ||
                !std::isfinite(x2) || !std::isfinite(y2))
                continue;
            x1 = max(0.0, min(x1, static_cast<double>(width)));
            y1 = max(0.0, min(y1, static_cast<double>(height)));
            x2 = max(0.0, min(x2, static_cast<double>(width)));
            y2 = max(0.0, min(y2, static_cast<double>(height)));
            if (x2 <= x1 || y2 <= y1)
                continue;
            const float left = static_cast<float>(x1);
            const float top = static_cast<float>(y1);
            const float right = static_cast<float>(x2);
            const float bottom = static_cast<float>(y2);
            if (right <= left || bottom <= top)
                continue;
            detections.push_back(YoloDetection{
                0, scores[idx], left, top, right, bottom});
        }
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const YoloDetection& det = detections[i];
            cout << "class=" << det.class_id
                 << " confidence=" << det.confidence
                 << " center=(" << (det.x1 + det.x2) * 0.5f
                 << ", " << (det.y1 + det.y2) * 0.5f << ")"
                 << " wh=(" << det.x2 - det.x1
                 << ", " << det.y2 - det.y1 << ")\n";
        }
        for (size_t i = 0; i < detections.size(); ++i)
            results[i] = detections[i];
        *out_count = static_cast<int>(detections.size());
        return 0;
    }
    catch (const exception& error)
    {
        cerr << "[Error] Postprocess failed: " << error.what() << '\n';
        return -1;
    }
    catch (...)
    {
        cerr << "[Error] Unknown postprocess error\n";
        return -1;
    }
}

extern "C" int yolo_cleanup(void)
{
    initialized = false;
    int result = 0;

    // Do not release tensors or the model while a task may still use them.
    if (task_handle != nullptr)
    {
        if (!task_completed)
        {
            if (hbDNNWaitTaskDone(task_handle, 0) != 0)
            {
                cerr << "[Error] Task completion unconfirmed; resources retained\n";
                return -1;
            }
            task_completed = true;
        }
        if (hbDNNReleaseTask(task_handle) != 0)
        {
            cerr << "[Error] Failed to release task; resources retained\n";
            return -1;
        }
        task_handle = nullptr;
        task_completed = false;
    }

    for (auto& tensor : output_tensors)
    {
        if (tensor.sysMem[0].virAddr != nullptr)
        {
            hbSysMem memory = tensor.sysMem[0];
            if (hbSysFreeMem(&memory) != 0)
            {
                cerr << "[Error] Failed to free output memory; record retained\n";
                result = -1;
            }
            else
            {
                tensor.sysMem[0] = hbSysMem{};
            }
        }
    }

    if (input_tensor.sysMem[0].virAddr != nullptr)
    {
        hbSysMem memory = input_tensor.sysMem[0];
        if (hbSysFreeMem(&memory) != 0)
        {
            cerr << "[Error] Failed to free input memory; record retained\n";
            result = -1;
        }
        else
        {
            input_tensor.sysMem[0] = hbSysMem{};
        }
    }

    if (result != 0)
        return -1;

    if (packed_handle != nullptr)
    {
        if (hbDNNRelease(packed_handle) != 0)
        {
            cerr << "[Error] Failed to release model; handle retained\n";
            return -1;
        }
        packed_handle = nullptr;
    }

    model_handle = nullptr;
    output_tensors.clear();
    input_tensor = hbDNNTensor{};
    input_width = input_height = 0;
    for (int i = 0; i < 6; ++i)
        output_order[i] = -1;

    return 0;
}
