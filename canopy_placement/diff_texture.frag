#version 430

uniform sampler2D tex_orig;
uniform sampler2D tex_optim;

in vec2 tex_pos;
out vec4 color;

void main()
{
	float orig = texture(tex_orig, tex_pos).r;
	float optim = texture(tex_optim, tex_pos).r;

	float diff = orig - optim;
	//float diff = orig;
	//color = vec4(vec3(diff * diff), 1.0f);
	color.r = diff * diff;
	//color.r = 1.5f;
}
