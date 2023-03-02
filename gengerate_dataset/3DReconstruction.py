import pyvista as pv
from pyvista import examples
import numpy as np
import vtk
from scipy import ndimage

pv.rcParams["use_panel"] = False

# 1.载入体素数据raw格式
imgData = np.fromfile('3d.raw', dtype=np.uint16)
imgData = imgData.reshape(821, 1206, 276) / 255.0  # 灰度调整到0~255,数据还原为原始长宽高
imgData = np.asarray(imgData, dtype=np.uint8)
# imgData = ndimage.zoom(imgData,0.5,order=3) #资源不够可考虑压缩


# 2.创建标准网格
grid = pv.UniformGrid()
# Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
grid.dimensions = np.array(imgData.shape) + 1
# Edit the spatial reference
grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

# 3.数据填充到网格
grid.cell_arrays["values"] = imgData.flatten(order="F")  # Flatten the array!

# 4.体素数据处理
threshold = grid.threshold([70, 255])  # 保留高灰度网格
surf = threshold.extract_geometry()  # 提取表面
smooth = surf.smooth(n_iter=200)  # 平滑表面

# 5.数据展示
p = pv.Plotter(point_smoothing=True)
p.set_background(color='black')
p.add_mesh(smooth, color="gray")
p.show(screenshot='shotcut.png')