#version 430

//layout(location=1) in vec3 normal;
//layout(location=2) in float scaled_height;

#define M_PI 3.1415926535897932384626433832795

in vec3 normal;
in float sheight;
in vec4 verts;
out vec3 color;

void main()
{
	
	/*
	color.r = sheight;
	color.g = sheight;
	color.b = sheight;
	*/

	//color = (normal + 1) / 2.0f;
	float angle = acos(abs(dot(normalize(normal), vec3(0.0f, 0.0f, -1.0f)))) / (0.5f * M_PI);
	color = vec3(angle);
	
	//color.r = color.g = color.b = 1.0f;
	//color.a = 1.0f;
}
