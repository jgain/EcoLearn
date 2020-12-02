#version 430

layout(location = 0)in vec3 vert;
layout(location = 1)in mat4 translate_mat;
layout(location = 5)in mat4 scale_mat;
layout(location = 9)in vec4 color_vec;
out vec3 outpos;
out float orig_z;
out vec3 abs_pos;
out vec4 color;

uniform mat4 ortho_mat;
//uniform mat4 translate_mat;
//uniform mat4 scale_mat;
uniform mat4 view_mat;

void main()
{
        vec4 vertmod = vec4(vert, 1.0f);	// create a vector from the vertex that can be multiplied for the MVP matrix
        abs_pos = (translate_mat * scale_mat * vertmod).xyz;	// absolute position on the landscape, before multiplying by view and projection matrices
        vec3 viewpos = (view_mat * translate_mat * scale_mat * vertmod).xyz;
        vertmod = ortho_mat * view_mat * translate_mat * scale_mat * vertmod;	// exact position on screen, for rendering purposes

        color = color_vec;
        //color.r = -viewpos.z / 100.0f;	// comment this out after debugging

	gl_Position = vertmod;

        outpos = vertmod.xyz;
}
