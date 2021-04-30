#version 330
layout (location = 0) in vec3 VertexPosition;
layout (location = 2) in vec2 VertexTexCoord;
layout (location = 3) in vec3 VertexNormal;
layout (location = 4) in float VertexDis;

out vec3 PosIn;
out vec3 PosNormal;
out vec2 TexCoordIn;
out vec4 PosLightSpaceIn;
out float DisIn;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main()

{
    vec4 viewPos = view * model * vec4(VertexPosition, 1.0);
    PosIn = viewPos.xyz; 
    TexCoordIn = VertexTexCoord;
	PosLightSpaceIn = lightSpaceMatrix * model * vec4(VertexPosition, 1);
	PosNormal = (transpose(inverse(view * model)) * vec4(VertexNormal, 1.0f)).xyz;
	PosNormal = normalize(PosNormal);
	DisIn = VertexDis;
	gl_Position = projection * viewPos;
}