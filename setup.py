from setuptools import setup


setup(name='modelutils',
      version='0.0.1',
      description='Utility scripts for various deep learning models',
      url='https://github.com/shonohs/modelutils',
      license='MIT',
      packages=['modelutils'],
      install_requires=[
          'coremltools',
          'numpy',
          'onnx',
          'onnxruntime',
          'Pillow',
          'tensorflow'
      ],
      entry_points={
          'console_scripts': [
              'modelutils-dump-coreml=modelutils.coreml.dump:main',
              'modelutils-dump-onnx=modelutils.onnx.dump:main',
              'modelutils-dump-tensorflow=modelutils.tensorflow.dump:main',
              'modelutils-load-onnx=modelutils.onnx.load:main',
              'modelutils-load-tensorflow=modelutils.tensorflow.load:main',
              'modelutils-run-caffe=modelutils.caffe.run:main',
              'modelutils-run-onnx=modelutils.onnx.run:main',
              'modelutils-run-tensorflow=modelutils.tensorflow.run:main',
              'modelutils-summarize-onnx=modelutils.onnx.summarize:main',
              'modelutils-summarize-tensorflow=modelutils.tensorflow.summarize:main'
          ]
      })
