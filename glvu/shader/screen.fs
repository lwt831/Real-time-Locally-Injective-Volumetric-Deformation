#version 330
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormalAndShadow;
uniform sampler2D gColor;
uniform sampler2D ssao;


uniform vec3 Ambient;
uniform vec3 LightColor;
uniform vec3 LightDirection;
uniform vec3 halfVector;
uniform float shininess;
uniform float strength;
uniform bool isShadowMap;
uniform bool isAo;

void main()
{             
    // Retrieve data from g-buffer
	vec2 texelSize = 1.0 / textureSize(gColor, 0);
	vec3 rgb = vec3(0.0);
	float Alpha = 0.0;
	/*ssaa (Super Sample Anti-aliasing)*/
	for(int i=0;i<2;i++)
		for(int j=0;j<2;j++)
		{
			vec2 TexCoord1 = TexCoord + vec2(i, j) * texelSize;
			//vec3 FragPos = texture(gPosition, TexCoord1).rgb;		//FragPos is used for point light source
			
			vec3 Normal = texture(gNormalAndShadow, TexCoord1).rgb;
			vec3 color = texture(gColor, TexCoord1).rgb;			
			float AmbientOcclusion = isAo ? texture(ssao, TexCoord1).r : 1;

			float shadow = isShadowMap ? texture(gNormalAndShadow, TexCoord1).w : 0.0;
			
			float diffuse = max(0.0,dot(Normal,LightDirection));
			float specular = max(0.0,dot(Normal,normalize(halfVector)));
			if (diffuse == 0.0)
			{
				specular = 0.0;
			}
			else
			{
				specular = pow(specular, shininess);
			}

			vec3 ambientLight = AmbientOcclusion * Ambient;
			vec3 diffuseLight = LightColor * diffuse;
			vec3 specularLight = LightColor * specular * strength;
			rgb += min(color*(ambientLight + (1.0 - shadow)*(diffuseLight + specularLight)),vec3(1.0,1.0,1.0));
			Alpha += texture(gColor, TexCoord1).w;
		}
	rgb /= 4.0;
	Alpha /= 4.0;
    FragColor = vec4(rgb,Alpha);
}