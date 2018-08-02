from setuptools import setup

setup(name='attrigram',
      version='0.1',
      description='A neural network for detecting independent class attributes.',
      url='http://github.com/brandontrabucco/attrigram',
      author='Brandon Trabucco',
      author_email='brandon@btrabucco.com',
      license='MIT',
      packages=['attrigram', 'attrigram.ops'],
      zip_safe=False)