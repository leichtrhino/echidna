#!/usr/bin/env python

from distutils.core import setup

setup(name='echidna',
      version='0.0.0',
      description='Apply deep clustering to several audio source separation models',
      author='LeichtRhino',
      author_email='leichtrhino@outlook.jp',
      url='https://github.com/leichtrhino/echidna',
      packages=[
          'echidna',
          'echidna.data',
          'echidna.models',
          'echidna.models.core',
          'echidna.models.multidomain',
          'echidna.metrics',
          'echidna.procs',
          'echidna.apps',
      ],
      )
