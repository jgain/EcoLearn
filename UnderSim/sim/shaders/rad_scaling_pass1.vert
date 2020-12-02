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
#version 150
#extension GL_ARB_explicit_attrib_location: enable

// vertex shader data: positions and texture coords only
layout (location=0) in vec3 vertex;
layout (location=1) in vec2 UV;

// transformations
uniform mat4 MV; // model-view mx
uniform mat4 MVproj; //model-view-projection mx
uniform mat3 normMx; // normal matrix

//colours and material
uniform vec4 surfColour;
uniform vec4 ptLightPos1; // for side wall lighting
uniform vec4 ptLightPos2;

// textures with height and precomputed normals
uniform sampler2D normalMap;
uniform sampler2D htMap;

uniform int drawWalls; // if 0, draw terrain, else drawing walls
uniform vec3 normalWall; // if drawing wall vertices, use this normal

uniform float terrainBase; // lowest point on terrain
uniform float terrainBasePad; // additional height padding to avoid 'thin' terrains

//varying vec3  normal;
//varying vec3  view;
//varying float depth;

// need for radiance scaling computations in FS
// these wil be interpolated per frag and used
// during render to FBO

out vec3 normal;
out float depth;
out vec3 view;
out vec4 colour;
out vec3 lightDir1; // for side wall lighting
out vec3 lightDir2;
out vec3 halfVector1;
out vec3 halfVector2;

// needed for contours
out vec3 pos;
out vec2 texCoord;

void main() {
    vec3 inNormal, v;

    texCoord = UV;
    v = vertex;

    // lookup normal in texture
    if (drawWalls == 0) // drawing ht field - lookup normals
    {
        inNormal = texture(normalMap, UV).xyz;
        // correct vertex position from heigt map...
        v.y = texture(htMap, UV).r;
    }
    else
    {
        inNormal = normalWall; // use fixed normal for all wall vertices
        if (UV.s < 0.0) // base of wall: must adapt to editing
            v.y = terrainBase - terrainBasePad;
        else
            v.y = texture(htMap, UV).r;
    }

    pos = v;

    // vertex in camera coords
    vec4 ecPos = MV * vec4(v, 1.0);

//  view   = -(gl_ModelViewMatrix*gl_Vertex).xyz;
//  normal = gl_NormalMatrix*gl_Normal;
//  depth  = log(-(gl_ModelViewMatrix*gl_Vertex).z);

  view   = -ecPos.xyz;
  //normal = normalize(inNormal);
  normal = normalize(normMx * inNormal);
  depth  = log(-ecPos.z);

  lightDir1  = normalize(ptLightPos1.xyz - ecPos.xyz);
  lightDir2  = normalize(ptLightPos2.xyz - ecPos.xyz);
  halfVector1 = normalize(normalize(-ecPos.xyz) + lightDir1);
  halfVector2 = normalize(normalize(-ecPos.xyz) + lightDir1);

  gl_Position = MVproj * vec4(v, 1.0); // clip space position

  colour = surfColour;
}
