#version 430

layout(location = 0) in vec4 vertex;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 incolor;
uniform mat4 mvp;
uniform vec3 eye;
uniform vec3 lightsource;

out vec4 color;

void main(void)
{
    vec3 lightray = vec3(vertex) - lightsource;
    gl_Position = mvp * vertex;
    //color = vec4(vec3(abs(dot(normalize(lightray), normal))), 1.0f);
    //color = vec4(abs(normal), 1.0f);
    color = vec4(incolor, 1.0f);
}
