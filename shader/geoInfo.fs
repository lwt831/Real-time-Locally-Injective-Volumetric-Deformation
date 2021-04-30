#version 330
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec4 gNormalAndShadow;	//gNormalAndShadow = vec4(Normal, IsShadow);
layout (location = 2) out vec4 gColor;	

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec4 fPosLightSpace;
in float fDis;

uniform sampler2D shadowMap;
uniform sampler2D Img;
uniform vec4 color;
uniform vec3 LightDirection;
uniform bool vizDis;
uniform bool vizTex;
uniform float textScale;


float ShadowCalculation(vec4 fragPosLightSpace, vec3 fNormal)
{
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	
	projCoords = projCoords * 0.5 + 0.5;
	float closestDepth = texture(shadowMap, projCoords.xy).w; 
	float currentDepth = projCoords.z;
	float bias = max(0.005 * (1.0 - dot(fNormal, LightDirection)), 0.0005);
	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
	for(int x = -1; x <= 1; ++x)
	{
		for(int y = -1; y <= 1; ++y)
		{
			if(projCoords.z > 1.0)
				shadow += 0.0;
			else
			{
				float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
				shadow += currentDepth - bias > pcfDepth ?  0.5f : 0.0f;      
			}			
		}    
	}
	shadow /= 9.0;
	return shadow;
}

void main()
{    
    gPosition = FragPos;
    vec3 fNormal = normalize(Normal);
	float shadow = ShadowCalculation(fPosLightSpace, fNormal);
	gNormalAndShadow = vec4(fNormal, shadow);
	vec4 ocolor = vizTex ? vec4((texture(Img, TexCoord/textScale).rgb), 1.0) : color;
	ocolor = vizDis ? vec4((texture(Img, vec2(fDis, 1)).rgb), 1.0) : ocolor;
	gColor = ocolor;
}