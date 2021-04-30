#version 330
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec4 gNormalAndShadow;	//gNormalAndShadow = vec4(Normal, IsShadow);
layout (location = 2) out vec4 gColor;	

uniform vec4 color;


void main()
{  
	if (length(gl_PointCoord - vec2(0.5, 0.5)) > 0.5) {
        discard;
    }  
	gColor = color;
	vec2 xy = (gl_PointCoord - vec2(0.5, 0.5))/0.5;
	float z = sqrt(1 - length(xy)*length(xy));
	gNormalAndShadow = vec4(xy, z, 0);

}