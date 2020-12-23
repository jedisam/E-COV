# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
spec_root = os.path.realpath(SPECPATH)

options = []
# options = [('v', None, 'OPTION')]
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
tf_hidden_imports = collect_submodules('tensorflow')
tf_datas = collect_data_files('tensorflow', subdir=None, include_py_files=True)

a = Analysis(['main.py'],
             pathex=['/home/jedi/Desktop/pls/e-cov2'],
             binaries=[],
             datas=tf_datas + [('shape_predictor_68_face_landmarks.dat','./face_recognition_models/models'),('shape_predictor_5_face_landmarks.dat','./face_recognition_models/models'),('mmod_human_face_detector.dat','./face_recognition_models/models'),('dlib_face_recognition_resnet_model_v1.dat','./face_recognition_models/models'),('face_detector/*', 'face_detector'), ('static/*', 'static')],
             hiddenimports=tf_hidden_imports + [],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
