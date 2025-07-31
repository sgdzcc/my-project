from maix import camera, display, image, nn, app, comm
import struct, os

report_on = True
APP_CMD_DETECT_RES = 0x02

def encode_objs(objs):
    """
    将目标检测结果编码为字节流，用于协议上报
    格式: 2B x + 2B y + 2B w + 2B h + 2B class_id + 4B score 
    """
    body = b''
    for obj in objs:
        body += struct.pack("<hhHHHf", obj.x, obj.y, obj.w, obj.h, obj.class_id, obj.score)
    return body

# 模型路径配置（根据实际文件位置调整）
model_path = "/root/models/model-229431.maixcam/model_229431.mud"  
if not os.path.exists(model_path):  
    print(f"警告: 模型文件 {model_path} 不存在，将尝试默认路径")
    # 可在此补充备用路径逻辑
else:  
    try:  
        detector = nn.YOLOv5(model=model_path)  
        print("模型加载成功！")
    except Exception as e:  
        print(f"错误: 加载模型失败 - {e}")
        # 加载失败时可尝试退出或补充备用模型逻辑
        exit(1)

# 初始化摄像头和显示
cam = camera.Camera(
    detector.input_width(), 
    detector.input_height(), 
    detector.input_format()
)
dis = display.Display()

# 初始化通信协议
p = comm.CommProtocol(buff_size=1024)

# 主循环
while not app.need_exit():
    # 1. 读取摄像头图像
    img = cam.read()
    if img is None:
        print("警告: 未读取到图像，跳过当前帧")
        continue

    # 2. 目标检测
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)

    # 3. 结果上报（可选）
    if len(objs) > 0 and report_on:
        body = encode_objs(objs)
        p.report(APP_CMD_DETECT_RES, body)

    # 4. 绘制检测结果（矩形框、文字、中心圆点）
    for obj in objs:
        # 绘制矩形框
        img.draw_rect(
            obj.x, obj.y, obj.w, obj.h, 
            color=image.COLOR_RED, 
            thickness=2
        )
        
        # 计算中心坐标
        center_x = obj.x + obj.w // 2
        center_y = obj.y + obj.h // 2
        
        # 绘制中心圆点（通过 thickness 实现填充效果）
        img.draw_circle(
            center_x, center_y, 
            radius=3,  # 圆点半径
            color=image.COLOR_BLUE, 
            thickness=3  # 线宽=半径实现填充
        )
        
        # 绘制类别与置信度文字
        msg = f"{detector.labels[obj.class_id]}: {obj.score:.2f}"
        img.draw_string(
            obj.x, obj.y - 10,  # 文字居上显示
            msg, 
            color=image.COLOR_YELLOW, 
            scale=2  # 文字放大更清晰
        )

    # 5. 显示最终图像
    dis.show(img)

# 程序退出清理（可选）
print("程序正常退出")
