import numpy as np
import numba
import time
import torch

# @numba.jit(nopython=True)

def depth_projection(depth):
    """
    depth: depth map(3D)
           x
           |
           |
           |_ _ _ y
          /
         /
        x
    功能：将深度补全出来的稠密深度图，从front view转成top view和side view
    点云范围，x轴 0-80。如果点云范围超过80m，则不考虑。
    如果通过深度补全得到的伪点云的值超过80米，则舍弃。
    """


    depth = torch.from_numpy(depth)

    # 对超出范围的深度进行裁剪
    depth[depth > 80] = 0

    # 初始化 top_view 和 top_view_depth_value 张量
    top_view = torch.zeros(1600, 1216, dtype=torch.long)
    top_view_depth_value = torch.zeros(1600, 1216, dtype=torch.float)

    # 对深度值进行非零检测
    non_zero_coords = torch.nonzero(depth)
    z_coords, y_coords = non_zero_coords[:, 0], non_zero_coords[:, 1]
    depth_values = depth[z_coords, y_coords]

    # 转换深度坐标
    x_coords = torch.floor(depth_values / 0.05).to(torch.long) - 1

    # 更新 top_view 和 top_view_depth_value
    top_view[x_coords, y_coords] = z_coords
    top_view_depth_value[x_coords, y_coords] = depth_values
    return top_view, top_view_depth_value
