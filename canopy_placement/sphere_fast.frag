#version 430

in vec3 outpos;
in vec3 abs_pos;
in vec4 color;
layout(location = 0)out vec4 frag_color;	//unique color of the tree we are rendering
layout(location = 1)out vec4 subtract_color;
//layout(location = 2)out vec4 diff_color;	// indicates whether the rebuilt CHM is above, level, or below the example CHM
layout(location = 3)out vec4 z_color;	// indicates the height above the ground of each point rendered
//layout(location = 9)in vec4 color;

//uniform vec4 color;	// unique color of the tree we are rendering
uniform sampler2D chm_texture;
uniform vec2 translate_limit;
uniform float chm_max_val;

void main()
{

        //subtract_color = vec4(0.0f);
        //subtract_color.a = 1.0f;

        //vec2 tex_loc = vec2(outpos.x / translate_limit.x, outpos.y / translate_limit.y);

        vec4 texcol = texture2D(chm_texture, (outpos.xy + 1) / 2);
        /*
        if (texcol.r > 70)
        {
            diff_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
        }
        else
        {
            diff_color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
        }
        */
        //texcol =  texcol / 55.0f;
        //diff_color = vec4((texcol.rgb * 0.3048 - abs_pos.z) / chm_max_val, 1.0f);
        //float val = diff_color.r;
        //diff_color.rgb = vec3(0.0f);
        //if (texcol.r == 0)	// no trees occur at that point in the CHM (green)
        //{
        //    diff_color.g = -val;
        //}
        //else if (val < 0)	// our tree is higher than CHM point (red)
        //{
        //    diff_color.r = -val;
        //    //subtract_color.a = 1.0f;
        //}
        //else	// CHM point is higher than our placed tree (blue)
        //{
        //    diff_color.b = val;
        //}
        //diff_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);

        frag_color = vec4(color.r, color.g, color.b, color.a);
        subtract_color = vec4(0.0f);
        subtract_color.a = 1.0f;		// this shader will only be run on locations where there are trees, so the subtract color must be zero (black)
        //frag_color = vec4(1.0, 1.0, 0.0, 1.0);

        z_color = vec4(vec3(abs_pos.z) / 255.0f, 1.0f);
		//z_color = vec4(vec3(1.0f), 1.0f);

}
