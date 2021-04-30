#version 330
layout (location = 0) out float oDepth;
layout (location = 1) out vec4 oColor;		

uniform vec4 color;
uniform vec3 Ambient;
uniform vec3 LightColor;
uniform vec3 LightDirection;
uniform vec3 halfVector;
uniform float shininess;
uniform float strength;

void main()
{  
	if (length(gl_PointCoord - vec2(0.5, 0.5)) > 0.5) {
        discard;
    }  
	vec2 xy = (gl_PointCoord - vec2(0.5, 0.5))/0.5;
	float z = sqrt(1 - length(xy)*length(xy));
	vec3 normal = vec3(xy, z);
	
	float diffuse = max(0.0,dot(normal, LightDirection));
	float specular = max(0.0,dot(normal, normalize(halfVector)));
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
	
	oDepth = 1.0;
	oColor = color * vec4((ambientLight + (diffuseLight + specularLight)),1.0);

}