#version 150

uniform vec4 specularCol;
uniform float shiny;
uniform vec4 matSpec;

uniform int drawWalls;
uniform int useRegionTexture;
uniform int drawContours;

uniform sampler2D overlayTexture;

in vec3 pos;
in vec2 texCoord;

in vec3 normal;
in vec3 halfVector;
in vec3 lightDir;
in vec4 diffuse;
in vec4 ambient;

out vec4 color;

void main(void)
{

    vec3 n, halfV,viewV,ldir;
    vec4 Diff;
    float NdotL,NdotHV;
    color = ambient; //global ambient
    //float att;

    // use textur emap for diffuse colour if terrain overlay is present

    Diff = (useRegionTexture == 1 ? texture(overlayTexture, texCoord) : diffuse);
    n = normalize(normal);

    //compute the dot product between normal and normalized lightdir

    NdotL = dot(n, normalize(lightDir));

    if (NdotL > 0.0) {

        //att = 1.0 / (gl_LightSource[0].constantAttenuation +
        //    gl_LightSource[0].linearAttenuation * dist +
        //    gl_LightSource[0].quadraticAttenuation * dist * dist);
        //color += att * (Diff * NdotL + ambient);
        color += Diff * NdotL;

        halfV = normalize(halfVector);
        NdotHV = max(dot(n,halfV), 0.0);
        //color += att * gl_FrontMaterial.specular * gl_LightSource[0].specular *
        //pow(NdotHV,gl_FrontMaterial.shininess);
        color += matSpec * specularCol * pow(NdotHV, shiny);
        }

    // draw contours

    if (drawContours == 1)
    {
        float f  = abs(fract (pos.y*70) - 0.5);
        float df = fwidth(pos.y*70);
        float g = smoothstep(-1.0*df, 1.0*df , f);

        float c = g;
        color = vec4(c,c,c,1.0) * color + (1-c)*vec4(1.0,0.0,0.0,1.0);
    }
}
