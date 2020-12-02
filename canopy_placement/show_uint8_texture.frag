#version 430

uniform sampler2D tex;

in vec2 tex_pos;
out vec4 color;

void main()
{
    color = vec4(texture(tex, tex_pos).r);
    color.a = 1.0f;
    //color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
