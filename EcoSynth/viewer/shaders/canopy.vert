#version 330
#extension GL_ARB_explicit_attrib_location: enable

// vertex shader: basicShader; simple Phong Model lighting

layout (location=0) in vec3 vertex;
layout (location=1) in vec2 UV;
layout (location=2) in vec3 vertexNormal;
layout (location=3) in mat4 iform;
layout (location=7) in vec4 coloff;

// transformations
uniform mat4 MV; // model-view mx
uniform mat4 MVproj; //model-view-projection mx
uniform mat3 normMx; // normal matrix

// per pixel values to be computed in fragment shader
out float alpha; // transparency of canopy

void main(void)
{
    vec3 v;

    v = vertex;
    alpha = coloff.a;
    gl_Position = MVproj * iform * vec4(v, 1.0); // clip space position
}
