//glsl version 4.5
#version 450

layout (location = 0) in vec3 inColor;

//output write
layout (location = 0) out vec4 outFragColor;

//push constants block
layout( push_constant ) uniform constants
{
	vec4 resolution_and_mouse;
	vec4 ticks;
	mat4 render_matrix;
} PushConstants;

void main()
{
    outFragColor = vec4(inColor, 1.0);
}

void main2() 
{
    vec2 resolution = PushConstants.resolution_and_mouse.xy;
    vec2 mouse_coords = PushConstants.resolution_and_mouse.zw;
    vec2 st = gl_FragCoord.xy/resolution;
    vec2 mt = mouse_coords/resolution;
    if (st.x < 0.2 || st.x > 0.8) {
        //outFragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (mt.x > 0.4 && mt.x < 0.6) {
        //outFragColor = vec4(0.0, 0.0, 1.0, 1.0);
    } else {
        float t = PushConstants.ticks.x;
        vec3 c;
	    float l, z = t;
	    for(int i=0; i<3; i++) {
		    vec2 uv, p=st.xy;
		    uv=p;
		    p-=.5;
		    p.x*=resolution.x/resolution.y;
		    z+=.07;
		    l=length(p);
		    uv+=p/l*(sin(z)+1.)*abs(sin(l*9.-z-z));
		    c[i]=.01/length(mod(uv,1.)-.5);
	    }
	    outFragColor=vec4(c/l,t);
	//     vec3 color = vec3(0.0);

    //     vec2 pos = vec2(0.5) - st;

    //     float r = length(pos) * 2.0;
    //     float a = atan(pos.y, pos.x);

    //     float f = cos(a*3.0);
    //     f = abs(cos(a*3.0));
        
    //     color = vec3(1.0 - smoothstep(f, f+0.02, r) );

    //     outFragColor = vec4(color, 1.0);
    }
}