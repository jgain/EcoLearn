#version 150
#extension GL_ARB_explicit_attrib_location: enable

// vertex shader: basicShader; simple Phong Model lighting

layout (location=0) in vec3 vertex;
layout (location=1) in vec2 UV;

// transformations
uniform mat4 MV; // model-view mx
uniform mat4 MVproj; //model-view-projection mx
uniform mat3 normMx; // normal matrix

//colours and material
uniform vec4 matDiffuse;
uniform vec4 matAmbient;
uniform vec4 lightpos; // in camera space
uniform vec4 diffuseCol;
uniform vec4 ambientCol;

uniform sampler2D normalMap;
uniform sampler2D htMap;

uniform int drawWalls; // if 0, draw terrain, else drawing walls
uniform vec3 normalWall; // if drawing wall vertices, use this normal

out vec3 normal; // vertex normal
out vec3 lightDir; // toLight
out vec3 halfVector;
out vec4 diffuse;
out vec4 ambient;

out vec3 pos;
out vec2 texCoord;

void main(void)
{
    vec3 inNormal, v;


    texCoord = UV;
    v = vertex;

    // lookup normal in texture
    if (drawWalls == 0) // drawing ht field - lookup normals
    {
        inNormal = texture(normalMap, UV).xyz;
        // correct vertex position from heigt map...
        v.y = texture(htMap, UV).r;

    }
    else
    {
        inNormal = normalWall; // use fixed normal for all wall vertices
        if (UV.s < 0.0) // base of wall: leave at z=0.0
            v.y = 0.0;
        else
            v.y = texture(htMap, UV).r;

    }

    pos = v;

    // map to camera space for lighting etc
    normal = normalize(normMx * inNormal);


    // vertex in camera coords
    vec4 ecPos = MV * vec4(v, 1.0);

    lightDir  = normalize(lightpos.xyz - ecPos.xyz);

    halfVector = normalize(normalize(-ecPos.xyz) + lightDir);

    diffuse = matDiffuse * diffuseCol;
    ambient = matAmbient * ambientCol;

    gl_Position = MVproj * vec4(v, 1.0); // clip space position
}
