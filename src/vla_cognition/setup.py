from setuptools import find_packages, setup

package_name = 'vla_cognition'

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
    maintainer='robo1',
    maintainer_email='ajgadgil6@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'reasoning_node = vla_cognition.reasoning_node:main',
            'safety_gate_node = vla_cognition.safety_gate_node:main',
        ],
    },
)
