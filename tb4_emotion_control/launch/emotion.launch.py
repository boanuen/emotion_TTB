"""Gazebo TB4 + node điều khiển bằng biểu cảm.
sim:=full  Gazebo đầy đủ (có GUI, có dock -> node tự undock; nặng, chậm trên WSL2)
sim:=lite  Gazebo nhẹ của tb4_voice_nav + cửa sổ GUI (không dock, nhanh hơn)
sim:=none  chỉ chạy node (Gazebo đã chạy sẵn)"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('turtlebot4_ignition_bringup'), 'launch', 'turtlebot4_ignition.launch.py'])),
        launch_arguments={'world': 'warehouse', 'slam': 'false', 'nav2': 'false', 'rviz': 'false'}.items(),
        condition=IfCondition(PythonExpression(["'", LC('sim'), "' == 'full'"])))
    lite = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('tb4_voice_nav'), 'launch', 'sim_lite.launch.py'])),
        condition=IfCondition(PythonExpression(["'", LC('sim'), "' == 'lite'"])))
    # cửa sổ Gazebo GUI cho chế độ lite (server headless); ogre1 vì WSL2 không chạy được ogre2.
    # Server cần LIBGL_ALWAYS_SOFTWARE=1 (lidar) nhưng GUI thì không: bỏ biến đó để GUI vẽ bằng GPU (D3D12),
    # đo được ~44% CPU thay vì ~144% khi vẽ bằng phần mềm. gpu:=false để quay lại vẽ phần mềm.
    lite_gui = lambda gpu: IfCondition(PythonExpression(
        ["'", LC('sim'), "' == 'lite' and '", LC('gpu'), "'.lower() == '", 'true' if gpu else 'false', "'"]))
    gui = TimerAction(period=30.0, actions=[
        ExecuteProcess(cmd=['env', '-u', 'LIBGL_ALWAYS_SOFTWARE', 'GALLIUM_DRIVER=d3d12',
                            'MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA',
                            'ign', 'gazebo', '-g', '--render-engine-gui', 'ogre'], condition=lite_gui(True)),
        ExecuteProcess(cmd=['ign', 'gazebo', '-g', '--render-engine-gui', 'ogre'], condition=lite_gui(False))])
    # chờ sim lên rồi mới nạp TensorFlow (nặng) để khỏi tranh CPU lúc Gazebo khởi động
    node = TimerAction(period=LC('delay'), actions=[Node(package='tb4_emotion_control', executable='emotion_control_node', output='screen',
                parameters=[{'cam': LC('cam'), 'width': LC('width'), 'calib': LC('calib'), 'undock': PythonExpression(["'", LC('sim'), "' != 'lite'"])}])])
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='full'),
        DeclareLaunchArgument('cam', default_value='0'),
        DeclareLaunchArgument('delay', default_value='50.0'),
        DeclareLaunchArgument('gpu', default_value='true'),    # GUI Gazebo vẽ bằng GPU (false = phần mềm)
        DeclareLaunchArgument('width', default_value='320'),   # 640 = nét hơn nhưng nặng hơn
        DeclareLaunchArgument('calib', default_value='3.0'),   # giây giữ mặt bình thường để hiệu chuẩn (0 = tắt)
        sim, lite, gui, node])
