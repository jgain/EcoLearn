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

//#version 120
//#extension GL_ARB_draw_buffers : enable
#version 150
#extension GL_ARB_explicit_attrib_location: enable

uniform int drawWalls;
uniform int drawContours;
uniform int drawGridLines;
uniform int drawWallContours;
uniform int drawWallGridLines;

uniform int useRegionTexture;
uniform int useConstraintTexture;

uniform int drawOutOfBounds;
uniform float outBoundsMax;
uniform float outBoundsMin;
uniform float outBoundsBlend;
uniform vec4 outBoundsCol;

uniform sampler2D overlayTexture;
uniform sampler2D constraintTexture;

in vec3 lightDir1; // for side wall lighting
in vec3 lightDir2;
in vec3 halfVector1;
in vec3 halfVector2;

in vec3  normal;
in vec3  view;
in float depth;
in vec4 colour;

in vec3 pos;
in vec2 texCoord;

layout (location = 0) out vec4 grad;
layout (location = 1) out vec4 norm;
layout (location = 2) out vec4 col;

// contour and grid line params
uniform float gridColFactor;
uniform float gridX;
uniform float gridZ;
uniform float gridThickness;
uniform float contourSep;
uniform float contourThickness;
uniform float contourColFactor;

void main(void) {
  const float eps = 0.01;
  const float foreshortening = 0.4;

  vec3 n = normalize(normal);

  float gs  = n.z<eps ? 1.0/eps : 1.0/n.z;

  gs = pow(gs,foreshortening);

  float gx  = -n.x*gs;
  float gy  = -n.y*gs;

  grad = vec4(gx,gy,depth,1.0);
  norm = vec4(n,gl_FragCoord.z);

  // do nota pply radiance scaling to walls
  if (drawWalls == 1)
     norm = vec4(0.0,0.0,0.0,gl_FragCoord.z);

  // use texture map for colour if terrain overlay is present

  col = (useRegionTexture == 1 ? texture(overlayTexture, texCoord) : colour);

  // blend in constraint texture if present:
  if (useConstraintTexture == 1)
  {
      vec4 texel = texture(constraintTexture, texCoord);
      col = vec4(mix(col.rgb, texel.rgb, texel.a), col.a);
  }

  if (drawWalls == 1) // compute ambient + diffuse term 2 opposite point lights
  {
      col = vec4(colour * ( clamp(dot(n, normalize(lightDir1) ), 0.0,1.0) +
                            clamp(dot(n, normalize(lightDir2) ), 0.0,1.0) + 0.3) );
  }

 // draw contours - always draw terrain contours if set, but do not necessarily draw side wall contours

  if (drawContours == 1 && (drawWalls == 0 || (drawWallContours == 1 && drawWalls == 1)) )
    {
        float f  = abs(fract (pos.y*contourSep) - 0.5);
        float df = fwidth(pos.y*contourSep);
        float g = smoothstep(contourThickness*df, 2.0*contourThickness*df , f);

        float c = g;
        //col = vec4(c,c,c,1.0) * col + (1-c)*vec4(1.0,0.0,0.0,1.0);
        col = vec4(c,c,c,1.0)*col + (1-c)*contourColFactor*col;
    }

   // draw grid lines, but not on walls
   if (drawGridLines == 1 && (drawWalls == 0 || (drawWalls == 1 && drawWallGridLines == 1)) )
   {
        vec2 f  = abs(fract (vec2(pos.x*gridX, pos.z*gridZ)) - 0.5);
        vec2 df = fwidth(vec2(pos.x*gridX, pos.z*gridZ));
        vec2 g = smoothstep(gridThickness*df, 2.0*gridThickness*df , f);
        float c = g.x * g.y;
        col = vec4(c,c,c,1.0)*col + (1-c)*gridColFactor*col;
   }

   // add in out of bounds colour indication
   if (drawOutOfBounds == 1 && drawWalls == 0 && (pos.y > outBoundsMax || pos.y < outBoundsMin) )
       col = mix(col, outBoundsCol, outBoundsBlend);

   col = clamp(col, 0.0, 1.0);
}
