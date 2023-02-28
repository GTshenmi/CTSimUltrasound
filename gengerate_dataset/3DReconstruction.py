import vtk
# arr3d 是你事先准备好的矩阵
# [矩阵转vtk类]
# # (这个函数就是这么长...我也很无奈)
# image = vtk.util.vtkImageImportFromArray.vtkImageImportFromArray()
# image.SetArray(arr3d)  # 加载三维矩阵
# image.SetDataSpacing(spacing)  # 设置PixelSpacing
# image.Update()

# 定义个图片读取接口
#读取PNG图片就换成PNG_Reader = vtk.vtkPNGReader()
Jpg_Reader = vtk.vtkJPEGReader()
Jpg_Reader.SetNumberOfScalarComponents(1)
Jpg_Reader.SetFileDimensionality(3)  # 说明图像是三维的
 # 定义图像大小，本行表示图像大小为（512*512*240）
Jpg_Reader.SetDataExtent(0, 512, 0, 512, 0, 240)
 # 设置图像的存放位置
Jpg_Reader.SetFilePrefix("D:/Code/ultrasound/python/ct sim ultrasound/gengerate_dataset/image/silce/S_CVH1_CVH01_")
 # 设置图像前缀名字
 #表示图像前缀为数字（如：0.jpg）
Jpg_Reader.SetFilePattern("%s%05d.jpg")
Jpg_Reader.Update()
Jpg_Reader.SetDataByteOrderToLittleEndian()

# 计算轮廓的方法
contour = vtk.vtkMarchingCubes()
contour.SetInputConnection(Jpg_Reader.GetOutputPort())
contour.ComputeNormalsOn()
contour.SetValue(0, 255)


mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(contour.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.SetBackground([0.1, 0.1, 0.5])
renderer.AddActor(actor)

window = vtk.vtkRenderWindow()
window.SetSize(512, 512)
window.AddRenderer(renderer)


interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

# 开始显示
window.Render()
interactor.Initialize()
interactor.Start()

# # [创建渲染算法]
# volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
# volumeMapper.SetInputData(Jpg_Reader.GetOutput())  # 加载渲染对象（原始提数据）
#
# # [创建物体颜色函数]
# colorFunc = vtk.vtkColorTransferFunction()  # 创建伪彩转换函数
# colorFunc.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
# colorFunc.AddRGBPoint(30.0, 196 / 255, 96 / 255, 0)
# colorFunc.AddRGBPoint(90.0, 254 / 255, 218 / 255, 182 / 255)
# colorFunc.AddRGBPoint(200.0, 239 / 255, 239 / 255, 239 / 255)
#
# # [创建物体不透明度函数]
# opacityFunc = vtk.vtkPiecewiseFunction()  # 创建分段函数
# opacityFunc.AddPoint(0, 0)
# opacityFunc.AddPoint(10.0, 0.01)
# opacityFunc.AddPoint(100.0, 0.5)
# opacityFunc.AddPoint(200, 1)
#
# # [创建物体属性]
# volumeProperty = vtk.vtkVolumeProperty()
# volumeProperty.SetColor(colorFunc)  # 设置颜色转换
# volumeProperty.SetScalarOpacity(opacityFunc)  # 设置不透明度
# volumeProperty.SetInterpolationTypeToLinear()  # 设置插值方案
# volumeProperty.ShadeOn()  # 阴影
#
# # [创建物体]
# vol = vtk.vtkVolume()
# vol.SetMapper(volumeMapper)  # 加载渲染算法
# vol.SetProperty(volumeProperty)  # 加载物体属性
#
# # [创建一个渲染器]
# ren = vtk.vtkRenderer()
# ren.AddVolume(vol)  # 加载物体
# ren.SetBackground(0.0, 0.0, 0.0)  # 设置背景颜色
#
# # [创建一个渲染窗口]
# renWin = vtk.vtkRenderWindow()
# renWin.AddRenderer(ren)  # 加载渲染器
# renWin.SetSize(600, 600)  # 设置窗口尺寸
#
# # [创建交互器]
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)  # 加载渲染窗口
#
# # [开始绘制]
# ren.ResetCamera()  # 重置相机
# renWin.Render()  # 渲染
# iren.Initialize()  # 初始化交互器
# iren.Start()  # show
