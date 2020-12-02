#version 430

in vec4 color;
layout (location = 0)out vec4 outcolor;

void main(void)
{
    if (gl_FrontFacing)
        outcolor = color;
    else
        outcolor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
}
