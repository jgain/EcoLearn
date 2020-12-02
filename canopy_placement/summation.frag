#version 430

uniform sampler2D tex_src;

in vec2 tex_pos;
out vec4 out_color;

void main()
{
	vec2 base_pos = tex_pos / 2;
	float sum = texture(tex_src, base_pos + vec2(0.5f, 0.0f)).r
			+ texture(tex_src, base_pos + vec2(0.0f, 0.5f)).r
			+ texture(tex_src, base_pos + vec2(0.5f, 0.5f)).r
			+ texture(tex_src, base_pos).r;
	out_color.r = sum;
	//out_color = vec4(sum, 1.0f);
	//out_color = vec4(texture(tex_src, base_pos).xyz, 1.0f);
	/*
	if (base_pos == vec2(0.0f, 0.0f))
		out_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	else
		out_color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
	*/
	//out_color = vec4(1.0, 0.0, 0.0, 1.0);
	//out_color = vec4(base_pos, 0.0f, 1.0f);
}
