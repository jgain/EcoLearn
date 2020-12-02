#version 330

#extension GL_ARB_explicit_attrib_location: enable

uniform vec4 specularCol;
uniform float shiny;
uniform vec4 matSpec;

uniform int drawWalls;
uniform sampler2D decalTexture;
uniform int useTexturing;

uniform sampler2D texs[64];

layout (location = 0) out vec4 grad;
layout (location = 1) out vec4 norm;
layout (location = 2) out vec4 color;

in vec3 normal;
in vec3 halfVector;
in vec3 lightDir;
in vec4 diffuse;
in vec4 ambient;

in vec2 texCoord;
flat in int texidx;

// NOTE: this shader does not compute a distance attentuation term for lighting.
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
        color += matSpec * specularCol * pow(NdotHV, shiny);

     }
     // decal texturing:
     if (useTexturing == 1)
     {
		if (texidx < 0)
		{
			vec4 texel = texture(decalTexture, texCoord);
			//// color = vec4(texCoord.x, texCoord.y, 0.0, 1.0);
			//// color = vec4(texel.r, texel.g, texel.b, color.a); // GL_REPLACE
			color = vec4(mix(color.rgb, texel.rgb, texel.a), color.a); // GL_DECAL
			//color = vec4(0.0f, 1.0f, 1.0f, 1.0f);
		}
		else
		{
			vec4 texcol = texture(texs[texidx], texCoord);
			if (texcol.a < 0.2f) discard;
			else color = texcol;
			/*
			if (texidx < 1)
				color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
			else if (texidx < 2)
				color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
			else if (texidx < 3)
				color = vec4(0.0f, 0.0f, 1.0f, 1.0f);
			else if (texidx < 4)
				color = vec4(0.0f, 1.0f, 1.0f, 1.0f);
			else
				color = vec4(0.8f, 0.8f, 0.8f, 1.0f);
			*/
		}

     }
     norm = vec4(0.0f, 0.0f, 0.0f, 0.0f);
}
