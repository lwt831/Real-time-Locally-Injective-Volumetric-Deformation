#version 330
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;


uniform int num_color_layer;

void main()

{
	vec2 texelSize = 1.0 / textureSize(tex0, 0);
	vec3 rgb = vec3(0.0);

	/*ssaa (Super Sample Anti-aliasing)*/
	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 2; j++)
		{
			vec2 TexCoord1 = TexCoord + vec2(i, j) * texelSize;

			
			float front_alpha = 0;
			float back_alpha = 0;
			float alpha = 0;
			
			alpha = texture(tex0, TexCoord1).a;
			back_alpha = front_alpha + (1 - front_alpha) * alpha;
			rgb += texture(tex0, TexCoord1).rgb * (1 - front_alpha) * alpha;
			
			front_alpha = back_alpha;
			alpha = texture(tex1, TexCoord1).a;
			back_alpha = front_alpha + (1 - front_alpha) * alpha;
			rgb += texture(tex1, TexCoord1).rgb * (1 - front_alpha) * alpha;
			
			front_alpha = back_alpha;
			alpha = texture(tex2, TexCoord1).a;
			back_alpha = front_alpha + (1 - front_alpha) * alpha;
			rgb += texture(tex2, TexCoord1).rgb * (1 - front_alpha) * alpha;
			/*for(int k = 0; k < num_color_layer; k++)
			{
				alpha = alpha + (1 - alpha) * texture(tex[k], TexCoord1).a;
				rgb += texture(tex[k], TexCoord1).rgb * alpha;		
			}*/				
		}
	rgb /= 4.0;
	FragColor = vec4(min(rgb,vec3(1.0, 1.0, 1.0)), 1.0);
}
