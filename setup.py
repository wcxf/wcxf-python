from setuptools import setup, find_packages

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(name='wcxf',
      version='1.6',
      author='David M. Straub, Jason Aebischer',
      author_email='david.straub@tum.de, jason.aebischer@tum.de',
      license='MIT',
      url='https://wcxf.github.io',
      description='Python API and command line interface for the Wilson Coefficient exchange format',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={
        'wcxf': ['data/*.yml',
                 'data/*.yaml',
                 'data/*.json',
                 'bases/*.json',
                 'bases/child/*.json',
                ]
      },
      install_requires=['pyyaml', 'ckmutil>=0.3.2', 'pandas',
                        'wilson'],
      extras_require={
            'testing': ['nose'],
      },
      entry_points={
        'console_scripts': [
            'wcxf = wcxf.cli:wcxf_cli',
            'wcxf2eos = wcxf.cli:eos',
            'wcxf2dsixtools = wcxf.cli:wcxf2dsixtools',
            'dsixtools2wcxf = wcxf.cli:dsixtools2wcxf',
            'wcxf2smeftsim = wcxf.cli:smeftsim',
        ]
      },
    )
