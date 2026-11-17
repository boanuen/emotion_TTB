from glob import glob
from setuptools import setup

pkg = 'tb4_emotion_control'
setup(
    name=pkg, version='0.1.0', packages=[pkg],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + pkg]),
        ('share/' + pkg, ['package.xml']),
        ('share/' + pkg + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='bao', maintainer_email='baodz20811@gmail.com',
    description='Facial emotion -> TurtleBot4 /cmd_vel', license='MIT',
    entry_points={'console_scripts': ['emotion_control_node = tb4_emotion_control.emotion_control_node:main']},
)
