import matplotlib.cm as cm


ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [ (0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [ (255, 0, 255), (255, 0, 0), (0, 255, 0)]



EndEffectorPts = [
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0, 0, 0.1, 1]
]

# RoboTwin/Agilex: link6 到 gripper 中心偏移约 0.085m (X方向)
# 根据 URDF: fl_joint7 origin xyz="0.08457 0.024493 -0.00010349"
Gripper2EEFCvt = [
    [1, 0, 0, 0.085],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

# 对于 LIBERO 单臂机器人，左右臂使用相同的转换
# AgiBotWorld 原始值: EEF2CamLeft = [0,0,-0.5236], EEF2CamRight = [0,0,0.5236]
EEF2CamLeft = [0,0,0]
EEF2CamRight = [0,0,0]