#version 150
#extension GL_ARB_explicit_attrib_location: enable

layout (location=0) in vec4 vertex;

void main( void )
{
    gl_Position = vertex;
}
