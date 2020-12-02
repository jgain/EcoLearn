#version 430

in vec2 texpos;
uniform sampler2D tex_in;

out vec4 color;

void main()
{
        color = vec4(texture2D(tex_in, texpos).rgb, 1.0f);
}
