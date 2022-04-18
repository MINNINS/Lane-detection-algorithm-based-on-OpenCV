import cv2
import numpy as np
from PIL import Image

# *******************工具函数*****************#
def calculate_slope(line):  # 计算线段line的斜率
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


def reject_abnormal_lines(lines, threshold):  # 离群值过滤
    # [1]计算出所有的斜率
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        # [2]计算斜率的平均值
        mean = np.mean(slopes)
        # [3]计算斜率与平均值的差值
        diff = [abs(s - mean) for s in slopes]
        # [4]找到差值最大的下标
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            # 5.5 删除掉这条线段
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines


def least_squares_fit(lines):  # 最小二乘拟合

    # 将lines中的线段拟合成一条线段
    # lines:线段集合,[np.array([[x_1,y_1,x_2,y_2]]),np.array([[x_1,y_1,x_2,y_2]])],...]
    # 线段上的两点,np.array([[xmin,ymin],[xmax,ymax]])

    # 取出所有坐标点
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # 进行直线拟合，得到多项式系数
    poly = np.polyfit(x_coords, y_coords, deg=1)  # 因为直线是一次曲线，所以deg=1
    # 根据多项式系数，计算两个直线上的点，用于唯一确定这条直线
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)


# -----end--#

# 主要函数： 对视频流进行处理
def show_lines(filename,speed):# 视频流文件，速度
    capture = cv2.VideoCapture(filename)
    while True:
        ret, color_img = capture.read()
        # 在这里color_img的类型不是str,即不是图片的链接，而是图片本身
        train_img = cv2.cvtColor(color_img, cv2.IMREAD_GRAYSCALE)
        # 1.Canny边缘检测
        edge_img = cv2.Canny(train_img, 33, 101)  # 可以调整上下阈值
        # 2.获取ROI区域 坐标：[2,369],[231,157],[339,92],[567,102],[750,240],[944,386]
        mask = np.zeros_like(edge_img)  # 获取一样的数组
        # 2.1得到掩码
        mask = cv2.fillPoly(mask, np.array([[[115, 686], [498, 368], [756, 341], [1178, 687]]]),
                            color=255)
        # 2.2ROI区域
        masked_edges_img = cv2.bitwise_and(edge_img, mask)  # 原图，掩码
        # 3.霍夫变换
        lines = cv2.HoughLinesP(masked_edges_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)
        # 按照斜率分成左右车道线
        left_lines = [line for line in lines if calculate_slope(line) > 0]
        right_lines = [line for line in lines if calculate_slope(line) < 0]
        # 4.离群值过滤
        left_lines = reject_abnormal_lines(left_lines, 0.2)
        right_lines = reject_abnormal_lines(right_lines, 0.2)
        # 5.最小二乘拟合
        left_line = least_squares_fit(left_lines)
        right_line = least_squares_fit(right_lines)
        # 6.绘制图片
        # img_02 = cv2.imread(color_img, cv2.IMREAD_COLOR)
        cv2.line(color_img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=5)
        cv2.line(color_img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=5)

        cv2.imshow('1', color_img)
        cv2.waitKey(speed)
