#version 150

uniform float terdim;
uniform int drawWalls;
uniform int drawCanopies;

// fragment shader: for sunlight
in vec2 texpos; // flat shaded colour per triangle used as lookup index
out vec4 color;

void main(void)
{
    int idx, r, g, b, x, y, dim;
    int roff, goff;
    
    color = vec4(1.0f, 1.0f, 1.0f, 1.0f); // background white
    
    // convert from uv position to colour assuming 8 bit resolution per colour channel
    if(drawWalls == 0 && drawCanopies == 0)
    {
        x = int(texpos.x * terdim);
        y = int(texpos.y * terdim);
        dim = int(terdim);
        idx = x * dim + y;
        r = idx / 65536;
        roff = r * 65536;
        g = (idx - roff) / 256;
        goff = g * 256;
        b = idx - roff - goff;
        color = vec4(float(r)/255.0f, float(g)/255.0f, float(b)/255.0f, 1.0f);
    }
}
