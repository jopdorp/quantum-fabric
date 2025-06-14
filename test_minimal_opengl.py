#!/usr/bin/env python3
"""
Minimal working OpenGL test with fallback to software rendering.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtOpenGL import QGLWidget, QGLFormat
from PyQt5.QtCore import QTimer
from OpenGL.GL import *

class MinimalOpenGLWidget(QGLWidget):
    def __init__(self):
        # Try to force software rendering if hardware fails
        format = QGLFormat()
        format.setDoubleBuffer(True)
        format.setDepth(True)
        format.setRgba(True)
        format.setAlpha(False)
        format.setSampleBuffers(False)  # Disable multisampling
        
        super().__init__(format)
        self.angle = 0
        
    def initializeGL(self):
        """Initialize OpenGL with minimal settings."""
        print("=== OpenGL Initialization ===")
        
        # Print OpenGL info
        try:
            vendor = glGetString(GL_VENDOR).decode('utf-8')
            renderer = glGetString(GL_RENDERER).decode('utf-8')
            version = glGetString(GL_VERSION).decode('utf-8')
            print(f"OpenGL Vendor: {vendor}")
            print(f"OpenGL Renderer: {renderer}")
            print(f"OpenGL Version: {version}")
        except Exception as e:
            print(f"Error getting OpenGL info: {e}")
        
        # Very basic setup
        glClearColor(0.2, 0.3, 0.4, 1.0)  # Steel blue background
        glEnable(GL_DEPTH_TEST)
        
        print("OpenGL initialization complete")
        
    def resizeGL(self, width, height):
        """Handle resize."""
        print(f"Resize: {width} x {height}")
        glViewport(0, 0, width, height)
        
    def paintGL(self):
        """Render scene."""
        # Clear with color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Very simple rendering test
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-2, 2, -2, 2, -2, 2)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(self.angle, 0, 0, 1)
        
        # Draw a simple colored square
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 0.0)  # Red
        glVertex2f(-0.5, -0.5)
        glColor3f(0.0, 1.0, 0.0)  # Green
        glVertex2f(0.5, -0.5)
        glColor3f(0.0, 0.0, 1.0)  # Blue
        glVertex2f(0.5, 0.5)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        glVertex2f(-0.5, 0.5)
        glEnd()
        
        self.angle += 1
        if self.angle >= 360:
            self.angle = 0

class MinimalMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal OpenGL Test")
        self.setGeometry(200, 200, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add status label
        self.status_label = QLabel("OpenGL Test - Should show rotating colored square")
        layout.addWidget(self.status_label)
        
        # Add OpenGL widget
        self.opengl_widget = MinimalOpenGLWidget()
        layout.addWidget(self.opengl_widget)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.opengl_widget.update)
        self.timer.start(50)  # 20 FPS

def main():
    # Try to force software rendering if needed
    if "--software" in sys.argv:
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        print("Forcing software OpenGL rendering")
    
    app = QApplication(sys.argv)
    
    print("=== Minimal OpenGL Test ===")
    print("This test renders a simple rotating colored square.")
    print("If you see nothing, try running with: python3 script.py --software")
    
    window = MinimalMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
