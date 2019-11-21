from setuptools import setup

setup(name='modelutils',
      version='0.1',
      description='Utility scripts for various deep learning models',
      url='https://github.com/shonohs/modelutils',
      license='MIT',
      packages=['modelutils'],
      install_requires=[
          'onnx',
          'onnxruntime',
          'Pillow',
      ],
      entry_points={
          'console_scripts': [
              'modelutils-dump-onnx=modelutils.onnx.dump:main',
              'modelutils-run-onnx=modelutils.onnx.run:main'
          ]
      })
