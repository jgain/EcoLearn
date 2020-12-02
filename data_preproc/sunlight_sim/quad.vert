#version 430

layout (location = 0) in vec3 pos;
uniform sampler2D tex_in;

out vec2 texpos;

void main()
{
        texpos = vec2((pos + 1) / 2);
        gl_Position = vec4(pos, 1.0f);
}
