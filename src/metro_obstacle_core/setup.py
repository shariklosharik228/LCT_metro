from setuptools import find_packages, setup

setup(name='metro_obstacle_core', version='1.0.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages',
                   ['resource/metro_obstacle_core']),
                  ('share/metro_obstacle_core', ['package.xml'])],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='Metro team', maintainer_email='team@example.com',
      description='Rail geometry and obstacle detection core.', license='Apache-2.0')
