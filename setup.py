import setuptools


setuptools.setup(name='modelutils',
                 version='0.1.0',
                 description='Utility scripts for various deep learning models',
                 url='https://github.com/shonohs/modelutils',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'coremltools',
                     'numpy',
                     'onnx',
                     'onnxruntime',
                     'Pillow',
                     'tensorflow-cpu'
                 ],
                 entry_points={
                     'console_scripts': [
                         'mudump=modelutils.commands.dump:main',
                         'muload=modelutils.commands.load:main',
                         'musummarize=modelutils.commands.summarize:main',
                         'murun=modelutils.commands.run:main',
                         'mugetdata=modelutils.commands.get_data:main',
                         'musetdata=modelutils.commands.set_data:main',
                         'npimage=modelutils.commands.image:main',
                         'npzeros=modelutils.commands.zeros:main',
                         'npdiff=modelutils.commands.diff:main',
                         'npjson=modelutils.commands.json:main',
                         'npget=modelutils.commands.get:main',
                         'npmax=modelutils.commands.max:main',
                         'npprint=modelutils.commands.print:main',
                         'nptranspose=modelutils.commands.transpose:main',
                         'modelutils-run-caffe=modelutils.caffe.run:main',
                         'modelutils-run-onnx=modelutils.onnx.run:main',
                         'modelutils-run-tensorflow=modelutils.tensorflow.run:main',
                         'modelutils-summarize-onnx=modelutils.onnx.summarize:main',
                         'modelutils-summarize-tensorflow=modelutils.tensorflow.summarize:main',
                         'modelutils-summarize-openvino=modelutils.openvino.summarize:main',
                         'modelutils-hash-onnx=modelutils.onnx.weights_hash:main',
                         'modelutils-hash-openvino=modelutils.openvino.weights_hash:main',
                     ]
                 })
