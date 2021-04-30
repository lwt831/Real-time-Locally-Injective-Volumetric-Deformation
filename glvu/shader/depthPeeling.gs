#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec3 PosIn[3];
in vec2 TexCoordIn[3];


out vec2 TexCoord;
out vec3 Normal;



void main()
{
	vec3 n = cross(PosIn[1].xyz-PosIn[0].xyz, PosIn[2].xyz-PosIn[0].xyz);
    for(int i = 0; i < gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position;
        TexCoord = TexCoordIn[i];
        Normal = n;
        EmitVertex();
    }
}