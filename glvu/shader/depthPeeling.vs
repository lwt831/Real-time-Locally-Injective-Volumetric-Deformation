#version 330
layout (location = 0) in vec3 VertexPosition;
layout (location = 2) in vec2 VertexTexCoord;

out vec3 PosIn;
out vec2 TexCoordIn;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()

{
    vec4 viewPos = view * model * vec4(VertexPosition, 1.0);
	PosIn = viewPos.xyz;
    TexCoordIn = VertexTexCoord;
	gl_Position = projection * viewPos;
}