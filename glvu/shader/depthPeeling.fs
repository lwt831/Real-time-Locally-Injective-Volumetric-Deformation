#version 330
layout (location = 0) out float oDepth;
layout (location = 1) out vec4 oColor;	
//out vec4 FragColor;
in vec2 TexCoord;
in vec3 Normal;

uniform sampler2D depthTexture;
uniform sampler2D Img;
uniform vec4 color;


uniform vec3 Ambient;
uniform vec3 LightColor;
uniform vec3 LightDirection;
uniform vec3 halfVector;
uniform float shininess;
uniform float strength;
uniform bool vizTex;
uniform int ith;

uniform float textScale;

float near_plane = 0.1;
float far_plane = 100.0;
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
}

void main()
{
	ivec2 wh = textureSize(depthTexture, 0);
	vec2 TexCoord1 = vec2(gl_FragCoord.x/wh.x, gl_FragCoord.y/wh.y);	//texcoord in front layer (depthTexture & colorTexture)
	
	//vec3 projCoords = gl_FragCoord.xyz / gl_FragCoord.w;
	//vec2 TexCoord1 = vec2(projCoords.x/wh.x, projCoords.y/wh.y);
	//projCoords = projCoords * 0.5 + 0.5;
	float dep = (gl_FragCoord.z);
	float bias = 0.00005;
	if(ith != 0)
    {
		if(LinearizeDepth(dep)/far_plane <= texture(depthTexture, TexCoord1.xy).r + bias)	//
			discard;
	}
		
	//phong reflection model
    vec3 fNormal = normalize(Normal);
	float diffuse = max(0.0,dot(fNormal, LightDirection));
	float specular = max(0.0,dot(fNormal, normalize(halfVector)));
	if (diffuse == 0.0)
	{
		specular = 0.0;
	}
	else
	{
		specular = pow(specular, shininess);
	}

	vec3 ambientLight = Ambient;
	vec3 diffuseLight = LightColor * diffuse;
	vec3 specularLight = LightColor * specular * strength;
	
	
	vec4 fcolor = vizTex ? vec4(texture(Img, TexCoord/textScale).rgb, 1.0) : color;	
	oDepth = LinearizeDepth(dep)/far_plane;
	oColor = fcolor * vec4((ambientLight + (diffuseLight + specularLight)),1.0);

}