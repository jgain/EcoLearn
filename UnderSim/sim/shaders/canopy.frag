#version 330

#extension GL_ARB_explicit_attrib_location: enable

uniform float terdim;
uniform int drawWalls;

in float alpha;
out vec4 color;

void main(void)
{
    color = vec4(0.0f, 0.0f, 0.0f, alpha); // background white
   
    // needs to blend only alpha channel with other colours
}
