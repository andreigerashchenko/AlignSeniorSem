# -*- mode: python ; coding: utf-8 -*-

import sys
import os

from kivy_deps import sdl2, glew
from kivymd import hooks_path as kivymd_hooks_path


path=os.path.abspath(".")
print(path)

block_cipher = None


a = Analysis(
    ['main_screen.py'],
    pathex=[path],
    binaries=[],
    datas=[],
    hiddenimports=['plyer.platforms.win.filechooser'],
    hookspath=[kivymd_hooks_path],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)	

a.datas += [('Code\main_screen.kv',path+'\\main_screen.kv','DATA')]
a.datas += Tree("env/Lib/site-packages/moviepy", prefix='moviepy')
a.datas += Tree("env/Lib/site-packages/imageio_ffmpeg/", prefix='imageio_ffmpeg')
a.datas += Tree("env/Lib/site-packages/kivy/data/glsl", prefix='data/glsl')

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_screen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe, Tree(path),
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_screen',
)
