#version 330
out float FragColor;
in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormalAndShadow;
uniform sampler2D texNoise;
uniform vec3 samples[64];
uniform mat4 projection;
uniform float model_scale;

int kernelSize = 64;
float radius = 0.5;
float bias = 0.005;

void main()
{
    vec3 fragPos = texture(gPosition, TexCoord).xyz;
    vec3 normal = normalize(texture(gNormalAndShadow, TexCoord).rgb);
	vec2 noiseScale = textureSize(gPosition, 0)/4.0;
    vec3 randomVec = normalize(texture(texNoise, TexCoord * noiseScale).xyz);
  
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
	
	radius = radius * model_scale;
    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; ++i)
    {
        vec3 sample = TBN * samples[i]; 
        sample = fragPos + sample * radius; 

        vec4 offset = vec4(sample, 1.0);
        offset = projection * offset; 
        offset.xyz /= offset.w; 
        offset.xyz = offset.xyz * 0.5 + 0.5; 

        float sampleDepth = texture(gPosition, offset.xy).z; 

        // range check & accumulate

        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= sample.z + bias ? 1.0 : 0.0) * rangeCheck;           
    }

    occlusion = 1.0 - (occlusion / kernelSize);
	
	float power = 1.0;
    FragColor = pow(occlusion, power);
}