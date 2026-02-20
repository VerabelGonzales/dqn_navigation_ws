from setuptools import find_packages, setup

package_name = 'dqn_robot_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='verabel',
    maintainer_email='abel.gonzales@ucb.edu.bo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_node = dqn_robot_nav.train_node:main',
            'test_node = dqn_robot_nav.test_node:main',
        ],
    },
)
