from setuptools import setup


_entry_points = '''\
[console_scripts]
trt=trt.__main__:cli
ptb=trt.tensorboard:cli
'''

setup(
    name='trt',
    version='1.0.0-rc1',
    author='Stoik',
    description='Collective training tools using native TensorFlow loop.',
    license='MIT',
    long_description='Collective training tools using native TensorFlow loop.',
    packages=['trt'],
    python_require='>3.6, <4.0.0',
    install_requires=[
        'absl-py>=0.7.0',
        'tensorflow>=2.0.0a0',
        'keras_applications>=1.0.6',
        'keras_preprocessing>=1.0.5',
        'numpy',
        'six',
        'pyyaml'
    ],
    entry_points=_entry_points,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Data Processing',
    ]
)
