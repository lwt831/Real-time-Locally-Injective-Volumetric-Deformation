#version 330 core
out float FragColor;
in vec2 TexCoord;
uniform sampler2D ssaoInput;
void main() 
{
    vec2 texelSize = 1.0 / vec2(textureSize(ssaoInput, 0));
    float result = 0.0;
	int radius = 2;  //blur kernel radius
    for (int x = -radius; x <= radius; ++x) 
    {
        for (int y = -radius; y <= radius; ++y) 
        {
            vec2 offset = vec2(x, y) * texelSize;
            result += texture(ssaoInput, TexCoord + offset).r;
        }
    }
    FragColor = result / float((radius * 2 + 1) * (radius * 2 + 1));
}  