#version 150

uniform vec4 specularCol;
uniform float shiny;
uniform vec4 matSpec;

uniform int drawWalls;

in vec2 texCoord;

in vec3 normal;
in vec3 halfVector;
in vec3 lightDir;
in vec4 diffuse;
in vec4 ambient;

out vec4 color;

// NOTE: this shader does not compute a disance attentuation term for lighting.
// some more variables need to be passed in for that.

void main(void)
{

    vec3 n, halfV,viewV,ldir;
    float NdotL,NdotHV;
    color = ambient; //global ambient
    //float att;

    n = normalize(normal);

    //compute the dot product between normal and normalized lightdir

    NdotL = max(dot(n, normalize(lightDir)), 0.0);

    if (NdotL > 0.0) {

        //att = 1.0 / (gl_LightSource[0].constantAttenuation +
        //    gl_LightSource[0].linearAttenuation * dist +
        //    gl_LightSource[0].quadraticAttenuation * dist * dist);
        //color += att * (diffuse * NdotL + ambient);
        color += diffuse * NdotL; // + ambient;

        halfV = normalize(halfVector);
        NdotHV = max(dot(n,halfV),0.0);
        //color += att * gl_FrontMaterial.specular * gl_LightSource[0].specular *
        //pow(NdotHV,gl_FrontMaterial.shininess);
        color += matSpec * specularCol * pow(NdotHV, shiny);
        }

}
