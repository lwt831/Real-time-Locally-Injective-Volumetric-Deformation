#version 330
layout (location = 0) in vec3 VertexPosition;
uniform mat4 lightSpaceMatrix;
uniform mat4 model;
void main()
{
	gl_Position = lightSpaceMatrix  * model * vec4(VertexPosition, 1.0f);
}