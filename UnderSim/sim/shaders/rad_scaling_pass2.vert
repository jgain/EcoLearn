/****************************************************************************
* Render Radiance Scaling                                                   *
* Meshlab's plugin                                                          *
*                                                                           *
* Copyright(C) 2010                                                         *
* Vergne Romain, Dumas Olivier                                              *
* INRIA - Institut Nationnal de Recherche en Informatique et Automatique    *
*                                                                           *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
*****************************************************************************
* Modified by P Marais, 2014                                                *
*****************************************************************************/
#version 150
#extension GL_ARB_explicit_attrib_location: enable

// pass through vertex shader: screen aligned quad
// vertex shader data: positions coords only

layout (location=0) in vec2 vertex;
layout (location=1) in vec2 tcoord;

out vec2 texCoord;

void main(void) {
  texCoord = tcoord;
  gl_Position = vec4(vertex.x, vertex.y, 0.0, 1.0); //clip space position - already in clip space
}
