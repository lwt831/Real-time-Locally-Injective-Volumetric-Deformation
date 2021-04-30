#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;


in vec3 PosIn[3];
in vec2 TexCoordIn[3];
in vec4 PosLightSpaceIn[3];
in vec3 PosNormal[3];
in float DisIn[3];


out vec3 FragPos;
out vec2 TexCoord;
out vec3 Normal;
out vec4 fPosLightSpace;
out float fDis;

uniform bool smooth_mode;




void main()
{
	vec3 n = cross(PosIn[1].xyz-PosIn[0].xyz, PosIn[2].xyz-PosIn[0].xyz);
    for(int i = 0; i < gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position;
        FragPos = PosIn[i];
        TexCoord = TexCoordIn[i];
        Normal = smooth_mode? PosNormal[i] : n;
		fDis = DisIn[i];
		fPosLightSpace = PosLightSpaceIn[i];
        EmitVertex();
    }
}