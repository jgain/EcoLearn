#version 430

uniform sampler2D chm_texture;

in vec2 tex_pos;
out vec4 color;

void main()
{
    color = vec4(texture(chm_texture, tex_pos).x / 255);  // why does it not return a value in the range 0 to 1??
    color.a = 1.0f;
    //color.r = color.r;
    //color.g = 0.0f;
    //color.b = 0.0f;
    //color.a = 1.0f;
}
