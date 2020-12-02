#version 430

layout(location = 0)in vec3 pos_in;
layout(location = 1)in vec2 tex_pos_in;
out vec2 tex_pos;

void main()
{
        tex_pos = tex_pos_in;
        gl_Position = vec4(pos_in.xyz, 1.0f);
}
