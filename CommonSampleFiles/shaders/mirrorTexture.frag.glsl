#version 400 
uniform mat4 view; 
uniform float lightIntensity;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform sampler2D mtexture;

in vec4  esNormal;
in vec4  esPosition;
in vec2  fragTexCoord;


out vec4 result;

void main()
{
	
	
	// Look up this particular fragments' diffuse texture color
	vec4 difTexColor = vec4(1.0);	
		difTexColor = texture( mtexture,  fragTexCoord);

	// Find:  Where is the light?
	vec4 esLightPos = view * vec4( lightPos, 1.0 );

	// Compute important vectors
    vec3 esNorm  =  normalize(esNormal.xyz);
	vec3 toEye   = -normalize(esPosition.xyz);
	vec3 toLight =  normalize(esLightPos.xyz-esPosition.xyz);
	
	// Compute a simple diffuse shading
	float NdotL  =  clamp(dot(toLight,esNorm),0,1)+1;
	//result = vec4( NdotL * lightIntensity * lightColor * difTexColor.xyz, 1.0 );
	result = difTexColor + vec4(0.2);
}