""" gen_texture.py
Create a texture png file
"""

import os
from PyQt6.QtGui import QImage, QColor

res_dir = os.path.dirname(os.path.realpath(__file__))
texture_red_path = os.path.join(res_dir, "tex_red.png")

def gen_texture():
    """Generate texture image for a plane (semi-transparent red)"""
    if not os.path.exists(texture_red_path):
        tex = QImage(4, 4, QImage.Format.Format_RGBA8888)
        tex.fill(QColor(255, 0, 0, 120))
        tex.save(texture_red_path)

if __name__ == "__main__":
    gen_texture()
