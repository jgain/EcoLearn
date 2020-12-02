#version 430

layout(location = 0)in vec2 tex_pos_in;
layout(location = 1)in vec3 pos_in;

out vec2 tex_pos;

void main()
{
	gl_Position = vec4(pos_in, 1.0f);
	tex_pos = tex_pos_in;
}
