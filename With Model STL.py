from GatorClient import GatorClient
from NatNetClient import NatNetClient
import vtk
import math
from vtk.util.numpy_support import numpy_to_vtk
colors = vtk.vtkNamedColors()
import quaternion
import time
import cv2
import numpy as np
from numpy.linalg import inv
import multiprocessing
from pyquaternion import Quaternion


def convert_to_color_vtk(image):
    img_VTK = vtk.vtkImageData()
    img_VTK.SetDimensions(image.shape[1], image.shape[0], 1)
    img_NumPyToVTK = numpy_to_vtk(np.flip(image.swapaxes(0,1),axis=1).reshape((-1,3), order='F'))
    img_NumPyToVTK.SetName('Image')
    img_VTK.GetPointData().AddArray(img_NumPyToVTK)
    img_VTK.GetPointData().SetActiveScalars('Image')
    return img_VTK


FiberHolder_motive_R = np.array(
    [[ 0.99597241, -0.00528292,  0.08950447],
    [-0.04271275,  0.84975324,  0.52544749],
    [-0.07883261, -0.52715418,  0.84610513]])

Camera_motive_R = np.array(
    [[-0.95725303,  0.14922037, -0.24779006],
    [-0.09631689,  0.64333137,  0.75950497],
    [ 0.27274473,  0.7509048 , -0.60145847]])

Cube_motive_R = np.array(
    [[ 0.93325606, -0.03363541,  0.35763358],
    [-0.15905936,  0.85398547,  0.49538766],
    [-0.32207645, -0.51920851,  0.79163709]])

Operator_motive_R= np.array(
    [[ 0.93358483, -0.04101839,  0.3560012 ],
    [-0.15324938,  0.85230724,  0.50008699],
    [-0.32393516, -0.52143059,  0.78941507]])


FiberHolder_bias = np.array([[-3.33333333], [1.5], [28.21]])
Cube_bias = np.array([0+55, -39.907+15 , 38.407 -5-29]).reshape(3, 1)
Camera_bias = np.array([[-6.9275 -2], [24.0975 +18], [10 ]])
Operator_bias = np.array([-85, -27.9075 , 15.4075 -7]).reshape(3, 1)

Operator_rel = np.zeros((4,4))
Operator_rel[3,3] = 1

Fiber_rel = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

'''Camera Parameters'''
degree_per_rad = 57.29578
depth_min = 10
depth_max = 1000
nx = 1280
ny = 720
coe = 1

fx = 1421.690042357781522
fy = 1421.545748740362114
fy = coe*fy
principal_point_x = 624.763189842383326
principal_point_y = 342.599546796083132


window_center_x = -2 * (principal_point_x - (nx / 2)) / nx
window_center_y = 2 * (principal_point_y - (ny / 2)) / ny

view_angle = degree_per_rad * (2.0 * math.atan2(ny / 2.0, fy))


fiber_angle_bias = math.atan2(113,-118)
fiber_rotation_m = np.array([[math.cos(fiber_angle_bias),-(math.sin(fiber_angle_bias))],
                             [math.sin(fiber_angle_bias), math.cos(fiber_angle_bias)]])


def receiveGatorFrame(pos2x, pos2y, pos2z, steaming_memory):
    steaming_memory[0] = -float(pos2x)
    steaming_memory[2] = float(pos2y)
    steaming_memory[1] = float(pos2z)

def receiveRigidBodyFrame(id, position, rotation, dataport):

    if id == 1:
        dataport[0] = position[0]*1000
        dataport[1] = position[1]*1000
        dataport[2] = position[2]*1000
        dataport[3] = rotation[0]
        dataport[4] = rotation[1]
        dataport[5] = rotation[2]
        dataport[6] = rotation[3]
    if id == 2:
        dataport[7] = position[0]*1000
        dataport[8] = position[1]*1000
        dataport[9] = position[2]*1000
        dataport[10] = rotation[0]
        dataport[11] = rotation[1]
        dataport[12] = rotation[2]
        dataport[13] = rotation[3]

    if id == 3:
        dataport[14] = position[0]*1000
        dataport[15] = position[1]*1000
        dataport[16] = position[2]*1000
        dataport[17] = rotation[0]
        dataport[18] = rotation[1]
        dataport[19] = rotation[2]
        dataport[20] = rotation[3]


def stream_Optitrack(data):
    streamingClient = NatNetClient()
    streamingClient.X = data
    streamingClient.rigidBodyListener = receiveRigidBodyFrame
    streamingClient.run()


def stream_Gator(steaming_memory):
    streamingGator = GatorClient()
    streamingGator.steaming_memory = steaming_memory
    streamingGator.SensorListener = receiveGatorFrame
    streamingGator.run()


def set_relative_Matrix(Fibermatrix, Operatormatrix, Optitrack_data, Gator_data):

    t0 = time.time()

    #Data Streaming
    cam_position_motive = np.array([Optitrack_data[0],Optitrack_data[1],Optitrack_data[2]]).reshape(3,1)
    operator_position_motive = np.array([Optitrack_data[7],Optitrack_data[8],Optitrack_data[9]]).reshape(3,1)
    fiber_holder_position_motive = np.array([Optitrack_data[14],Optitrack_data[15],Optitrack_data[16]]).reshape(3,1)
    print(Gator_data[0])
    fiberXY = np.array([[Gator_data[0]], [Gator_data[1]]])
    fiberXY = np.dot(fiber_rotation_m, fiberXY)

    fibertip_position_gator = np.array([fiberXY[0], fiberXY[1], Gator_data[2]+40]).reshape(3,1)
    fibertip_to_holder_position_sw = fibertip_position_gator + FiberHolder_bias


    camera_rotation_qua = Quaternion(Optitrack_data[6], Optitrack_data[3], Optitrack_data[4], Optitrack_data[5])
    camera_rotation_motive = camera_rotation_qua.rotation_matrix

    operator_rotation_qua = Quaternion(Optitrack_data[13], Optitrack_data[10], Optitrack_data[11], Optitrack_data[12])
    operator_rotation_motive = operator_rotation_qua.rotation_matrix

    fiberholder_rotation_qua = Quaternion(Optitrack_data[20], Optitrack_data[17], Optitrack_data[18], Optitrack_data[19])
    fiberholder_rotation_motive = fiberholder_rotation_qua.rotation_matrix

    camera_model_orientation = np.dot(camera_rotation_motive, Camera_motive_R)
    operator_model_orientation = np.dot(operator_rotation_motive, Operator_motive_R)
    fiberholder_orientation = np.dot(fiberholder_rotation_motive, FiberHolder_motive_R)

    fibertip_position_motive = np.dot(fiberholder_orientation, fibertip_to_holder_position_sw)
    operator_bias_real = np.dot(operator_model_orientation, Operator_bias)
    Camera_bias_real = np.dot(camera_model_orientation, Camera_bias)

    operator_relative_position = operator_position_motive - cam_position_motive + operator_bias_real - Camera_bias_real

    camera_model_view = inv(camera_model_orientation)

    operator_relative_translation = np.dot(camera_model_view, operator_relative_position)
    operator_relative_orientation = np.dot(camera_model_view, operator_model_orientation)

    fibertip_relative_translation = np.dot(camera_model_view, (fibertip_position_motive +fiber_holder_position_motive- cam_position_motive - Camera_bias_real))
    print(fibertip_relative_translation)
    Operator_rel[0:3,3] = operator_relative_translation.reshape(1,3)
    Operator_rel[0:3,0:3] = operator_relative_orientation
    Fiber_rel[0:3,3] = fibertip_relative_translation.reshape(1,3)

    Fibermatrix.DeepCopy(tuple(np.ravel(Fiber_rel)))
    Operatormatrix.DeepCopy(tuple(np.ravel(Operator_rel)))

    distances_bewteen_tipandoperator = np.linalg.norm(fibertip_relative_translation - operator_relative_translation)
    return operator_relative_translation, distances_bewteen_tipandoperator


class BackGroundRefresh():
    def __init__(self):
        self.timer_count = 0


    def execute(self, obj, event):
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = convert_to_color_vtk(rgb_frame)
        self.image_actor.SetInputData(img)

        translation, reladis = set_relative_Matrix(fiber_RT, operator_RT, Optitrack_data, Gator_data)

        self.actor.SetUserMatrix(operator_RT)
        self.organ_actor.SetUserMatrix(fiber_RT)
        self.sephere_actor.SetUserMatrix(fiber_RT)

        self.navigation_text.SetInput("The Distance Between Operator and Suedo is %d"%(reladis))
        self.navigation_information.SetMapper(self.navigation_text)

        iren = obj
        iren.GetRenderWindow().Render()

        self.timer_count += 1


if __name__ == '__main__':
    '''Start Optitrack Data streaming'''
    Optitrack_data = multiprocessing.Array('d', [0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0])
    Optitrack_streaming = multiprocessing.Process(target=stream_Optitrack, args=(Optitrack_data,))
    Optitrack_streaming.start()

    Gator_data = multiprocessing.Array('d', [0, 0, 0])
    Gator_streaming = multiprocessing.Process(target=stream_Gator, args=(Gator_data,))
    Gator_streaming.start()


    '''Set Webcam and cap frame from webcam'''
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = convert_to_color_vtk(rgb_frame)

    reader = vtk.vtkSTLReader()
    reader.SetFileName('Operator.STL')

    '''Setting Operator Model'''
    reader = vtk.vtkSTLReader()
    reader.SetFileName('Operator.STL')
    transform = vtk.vtkTransform()
    transform.Translate(0, -3, -3)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    Operator_mapper = vtk.vtkPolyDataMapper()
    Operator_mapper.SetInputConnection(transformFilter.GetOutputPort())
    Operator_actor = vtk.vtkActor()
    Operator_actor.SetMapper(Operator_mapper)
    Operator_actor.GetProperty().SetColor(colors.GetColor3d("SteelBlue"))
    Operator_actor.GetProperty().SetOpacity(0.5)

    scene_renderer = vtk.vtkRenderer()
    scene_renderer.AddActor(Operator_actor)
    scene_renderer.SetLayer(1)
    scene_renderer.InteractiveOff()


    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(0, 0, 0)
    sphereSource.SetRadius(5)
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphereSource.GetOutputPort())
    sephere_actor = vtk.vtkActor()
    sephere_actor.SetMapper(sphere_mapper)
    sephere_actor.GetProperty().SetColor(colors.GetColor3d("AliceBlue"))
    scene_renderer.AddActor(sephere_actor)

    organ_reader = vtk.vtkSTLReader()
    organ_reader.SetFileName('Organ.stl')
    o_transform = vtk.vtkTransform()
    o_transform.Translate(64.75, -153.25, 572.75)
    o_transformFilter = vtk.vtkTransformPolyDataFilter()
    o_transformFilter.SetInputConnection(organ_reader.GetOutputPort())
    o_transformFilter.SetTransform(o_transform)
    o_transformFilter.Update()
    organ_mapper = vtk.vtkPolyDataMapper()
    organ_mapper.SetInputConnection(o_transformFilter.GetOutputPort())
    organ_actor = vtk.vtkActor()
    organ_actor.SetMapper(organ_mapper)
    organ_actor.GetProperty().SetColor(colors.GetColor3d("Maroon"))
    organ_actor.GetProperty().SetOpacity(0.4)
    scene_renderer.AddActor(organ_actor)


    '''Navigation Information'''
    singleLineTextProp = vtk.vtkTextProperty()
    singleLineTextProp.SetFontSize(24)
    singleLineTextProp.SetFontFamilyToArial()
    singleLineTextProp.BoldOff()
    singleLineTextProp.ItalicOff()
    singleLineTextProp.ShadowOff()

    Navigation_Text = vtk.vtkTextMapper()
    Navigation_Text.SetInput("Navigation Information")
    tprop = Navigation_Text.GetTextProperty()
    tprop.ShallowCopy(singleLineTextProp)
    tprop.SetVerticalJustificationToBottom()

    navigation_information = vtk.vtkActor2D()
    navigation_information.SetMapper(Navigation_Text)
    navigation_information.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    navigation_information.GetPositionCoordinate().SetValue(0.05, 0.85)
    scene_renderer.AddActor2D(navigation_information)

    '''Set camera background'''
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(img)
    background_renderer = vtk.vtkRenderer()
    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()
    background_renderer.AddActor(image_actor)

    '''Set background camera configuration'''
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    extent = img.GetExtent()
    camera_back = vtk.vtkCamera()
    camera_back.ParallelProjectionOn()
    xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera_back.GetDistance()
    camera_back.SetParallelScale(0.5 * yd)
    camera_back.SetFocalPoint(xc, yc, 0.0)
    camera_back.SetPosition(xc, yc, d)
    background_renderer.SetActiveCamera(camera_back)

    '''After build two layers' actor, configure them into renderwindow'''
    render_window = vtk.vtkRenderWindow()
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(scene_renderer)
    render_window.AddRenderer(background_renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    '''Set camera for real camera'''
    camera = vtk.vtkCamera()
    camera.SetPosition(0, 0, 0)
    camera.SetFocalPoint(0, 0, 1)
    camera.SetViewUp(0, 1, 0)
    camera.SetClippingRange(depth_min, depth_max)
    camera.SetWindowCenter(window_center_x, window_center_y)
    camera.SetViewAngle(view_angle)

    scene_renderer.SetActiveCamera(camera)
    render_window.Render()

    render_window_interactor.Initialize()
    render_window.SetDeviceIndex(0)
    render_window.SetDesiredUpdateRate(60)
    render_window.SetSize(1280, 720)

    fiber_RT = vtk.vtkMatrix4x4()
    operator_RT = vtk.vtkMatrix4x4()
    bgr = BackGroundRefresh()
    bgr.image_actor = image_actor
    bgr.actor = Operator_actor
    bgr.camera = camera
    bgr.frame = frame
    bgr.organ_actor = organ_actor
    bgr.sephere_actor = sephere_actor
    bgr.navigation_information = navigation_information
    bgr.navigation_text = Navigation_Text
    bgr.fiber_RT = fiber_RT
    bgr.operator_RT = operator_RT

    render_window_interactor.AddObserver('TimerEvent', bgr.execute)
    render_window_interactor.CreateRepeatingTimer(1)
    render_window_interactor.Start()


