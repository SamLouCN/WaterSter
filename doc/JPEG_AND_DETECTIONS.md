# JPEG 预览与 YOLO 坐标

在 WaterSter 根目录启动：

```sh
python3 src/main.py
```

该命令启动 Nginx 和受监控的 YOLO 后端。预览默认使用本项目
`lib/libmultimedia.so` 的 X5 JPU JPEG 编码接口，不再调用 OpenCV 软件
`imencode`。`src/jpeg_encoder.c` 是 C ABI 封装；Python 首次使用时自动编译为
`lib/libwaterster_jpeg.so`，需要板上已有的 `cc` 和 `include/` 头文件。
编码器按任务复用，停止或失败时释放。初始化/编码失败会记录错误，由现有
进程监控器重启后端，不会静默切回软件编码。

预览路径：相机 MJPEG → 硬件解码 NV12 → YOLO → 发布坐标 → NV12 转 BGR
→ 绘框 → 转 I420 并调整为 JPEG 全范围颜色 → JPU 编码 → 发布 JPEG。
JPU 输入要求宽度为 16 的倍数、高度为 8 的倍数，默认 640×480 满足要求。
编码质量为 85。C 封装负责复制 DMA 输入与 JPEG 输出并归还输出缓冲区。

只需坐标时：

```sh
python3 src/main.py --detections-only
curl http://192.168.127.10/detections
```

此模式保留相机硬件解码与 YOLO，完全跳过绘制、预览颜色转换、编码器初始化
和 JPEG 编码。Nginx、`/health`、`/detections` 仍可用；图片和视频流端点
立即返回 HTTP 503，并提示使用 `/detections`。

`/detections` 在两种模式下都返回最新推理结果，独立于该帧 JPEG 是否生成。
结果包含 `run`、`sequence`、`source_sequence`、`width`、`height`、
`detections`、`detection_count`、`source_to_result_ms`、`age_seconds`、
`running` 和 `error`。`detections` 内含 `class_id`、`confidence`、`xyxy`、
`center` 和 `wh`，坐标单位为相机原图像素。无目标时返回空数组。
结果尚未就绪、超过 3 秒、任务停止或发生错误时返回 HTTP 503。

Python 同进程消费者可调用 `task.wait_result(previous, timeout)` 等待新结果，
或 `task.result()` 获取快照；返回对象可独立修改。时间戳使用板端
`time.monotonic()`，不代表 UTC。进程重启后 `run` 改变、序号重新计数。

默认预览模式中，当前帧坐标先于 JPEG 交付，但同一个 YOLO 工作线程仍需完成
预览后才能处理下一帧。要完全移除预览对处理吞吐量的影响，请使用
`--detections-only`。

`logs/yolo_test.log` 新增 `task_result`，记录坐标就绪时间。
`task_frame` 保留预览耗时，增加 `jpeg_backend`、`source_to_result_ms`、
`result_publish_ms`、`bgr_to_i420_ms`、`jpeg_encoder_init_ms`。
`jpeg_encode_ms` 包含 C 调用、DMA 输入复制、硬件等待和 JPEG 输出复制；
`jpeg_bytes_ms` 是转成 Python bytes 的额外复制。编码器首次初始化单独计时。
`source_to_result_ms` 与 `source_to_publish_ms` 从相机完成硬件解码后开始计算，
不包含传感器曝光、USB 传输、相机解码或浏览器显示延迟。

验证命令：

```sh
PYTHONPATH=src python3 -m unittest yolo_test_task.test_pipeline yolo_test_task.test_timing yolo_test_task.test_supervisor -v
```

实际板端集成测试已验证 Nginx 转发、JPEG 解码尺寸、MJPEG 帧头、连续坐标更新、
只取坐标模式无 JPEG、退出资源释放，以及 640×480 / 1280×720 色块编码与颜色范围。
