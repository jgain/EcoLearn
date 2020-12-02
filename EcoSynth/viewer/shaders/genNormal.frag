#version 150
#extension GL_ARB_explicit_attrib_location: enable

// fragment shader: gennormal
out vec4 norm;

uniform vec2 imgSize;
uniform sampler2D htMap;

uniform vec2 scale;

// NB!!! y and z have been swapped: y is now height above (x,z) base plane.
// F = (x, f(x,z), z); need dF/dx and dF/dz to define base vectors for cross product
// dF/dx = (1, df/dx,0); dF/dz = (0, df/dz, 1)

void main(void)
{
    vec2  pos, delta;

    // const float scale = 10.0*1024.0; // should be passed as parameter

    vec3 dfdx = vec3(1.0, 0.0, 0.0), // df/dx and df/dy used to define normal
         dfdy = vec3(0.0, 0.0, 1.0); // y component filled in below

    delta.x = 1.0f/imgSize.x;
    delta.y = 1.0f/imgSize.y;

    pos = (gl_FragCoord.xy + 0.5 ) / imgSize.xy;

    dfdx.y = (texture(htMap, vec2(pos.x+delta.x, pos.y)).r/scale.x -
              texture(htMap, vec2(pos.x-delta.x, pos.y) ).r/scale.x)/(2.0*delta.x);
    dfdy.y = (texture(htMap, vec2(pos.x, pos.y+delta.y) ).r/scale.y -
              texture(htMap, vec2(pos.x, pos.y-delta.y) ).r/scale.y)/(2.0*delta.y);


    /*

    pos = gl_FragCoord.xy;

    ivec2 p = ivec2(pos);

    dfdx.z = (texelFetch(htMap, ivec2(p.x+1, p.y), 0 ).r -
              texelFetch(htMap, ivec2(p.x-1, p.y), 0 ).r)/(0.004);
    dfdy.z = (texelFetch(htMap, ivec2(p.x, p.y+1), 0 ).r -
              texelFetch(htMap, ivec2(p.x, p.y-1), 0 ).r)/(0.004);
    */

    vec3 n = -cross(dfdx, dfdy);

//  n = vec3(0.0, 1.0, 0.0);
    norm = vec4(normalize(n), 0.0);
}
