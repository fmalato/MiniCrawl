from distutils.core import setup

setup(
    name='minicrawl',
    packages=['minicrawl'],
    version='0.1',
    license='Apache-2.0',
    description='MiniCrawl is a minimalistic dungeon crawler-like 3D environment for Reinforcement and Imitation Learning research.',
    author='Federico Malato',
    author_email='federico.malato@uef.fi',
    url='https://github.com/fmalato/MiniCrawl',
    download_url='https://github.com/fmalato/MiniCrawl/archive/refs/tags/v0.1.tar.gz',
    keywords=['Environment', 'Reinforcement', 'Imitation', 'Learning', 'Robotics', 'Gym'],
    python_requires=">=3.7, <3.11",
    install_requires=[
        'gymnasium',
        'miniworld',
        'numpy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache 2.0',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
