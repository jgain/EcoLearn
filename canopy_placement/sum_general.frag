#version 430

uniform sampler2D tex_src;

uniform vec2 source_size;
uniform vec2 target_size;

in vec2 tex_pos;
out vec4 out_color;

void main()
{
    vec2 ratio = target_size / source_size;
    vec2 src_rel_pos = tex_pos * ratio;

    vec3 sum = texture(tex_src, src_rel_pos).xyz;
    //vec3 sum = texture(tex_src, tex_pos).xyz;
    if (src_rel_pos.x + 0.5f <= 1.0f)
    {
        sum += texture(tex_src, src_rel_pos + vec2(0.5f, 0.0f)).xyz;
    }

    if (src_rel_pos.y + 0.5f <= 1.0f)
    {
        sum += texture(tex_src, src_rel_pos + vec2(0.0f, 0.5f)).xyz;
    }

    if (src_rel_pos.x + 0.5f <= 1.0f
            && src_rel_pos.y + 0.5f <= 1.0f)
    {
        sum += texture(tex_src, src_rel_pos + vec2(0.5f, 0.5f)).xyz;
    }

	out_color = vec4(sum, 1.0f);
    //out_color = vec4(1.0, 0.0, 0.0, 1.0);
}
