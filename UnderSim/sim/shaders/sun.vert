#version 150
#extension GL_ARB_explicit_attrib_location: enable

// vertex shader: for sunlight, simple indexed colour per triangle, passed through directly

layout (location=0) in vec3 vertex;
layout (location=1) in vec2 UV;

// transformations
uniform mat4 MV; // model-view mx
uniform mat4 MVproj; //model-view-projection mx
uniform mat3 normMx; // normal matrix

uniform sampler2D htMap; // height values as texture
uniform sampler2D normalMap; // normal values as texture

uniform int drawWalls; // if 0 draw terrain, else drawing walls
uniform int drawCanopies; // if 0 draw terrain, else drawing walls

// planar position is converted to a colour in the vertex shader
out vec2 texpos;

void main(void)
{
    vec3 v;
    
    v = vertex;
    texpos = UV;
    
    if (drawWalls == 0) // drawing ht field - lookup normals
    {
        v.y = texture(htMap, UV).r; // correct vertex position from heigt map
        
    }
    else
    {
        if (UV.s < 0.0) // base of wall: leave at z=0.0
            v.y = 0.0;
        else
            v.y = texture(htMap, UV).r;
    }
    gl_Position = MVproj * vec4(v, 1.0); // clip space position
}
