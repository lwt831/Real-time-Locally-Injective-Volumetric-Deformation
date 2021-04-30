#version 330
layout (location = 0) in vec3 VertexPosition;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()

{
    vec4 viewPos = view * model * vec4(VertexPosition, 1.0);
	gl_Position = projection * viewPos;
}