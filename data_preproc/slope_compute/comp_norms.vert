#version 430

layout(location=0) in vec3 in_verts;
layout(location=1) in vec3 normal_in;
layout(location=2) in float scaled_height;
out vec4 verts;
out vec3 normal;
out float sheight;

uniform mat4 mvmatrix;

void main()
{
	gl_Position = (mvmatrix * vec4(in_verts, 1.0f));
	sheight = scaled_height;
	normal = normal_in;
	//gl_Position = vec4(in_verts, 1.0f);
	//verts = gl_Position;
}
