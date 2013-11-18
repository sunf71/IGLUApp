#version 400

uniform mat4 view; 
uniform float lightIntensity;

uniform samplerBuffer  matlInfoTex;
uniform sampler2DArray matlTextures;

in vec4  esNormal;
in vec4  esPosition;
in vec2  fragTexCoord;
in float fragMatlID;

out vec4 result;

const vec3 wslightPos = vec3( 2.13, 5.487, 2.27 );
const vec3 lightColor = vec3( 1.0, 0.85, 0.43); //vec3( 18.4, 15.6, 8.0 );

// This is the actual code that is executed.
void main( void )
{
	// Look up data about this particular fragment's material
	int matlIdx     = 4 * int(fragMatlID+0.5);
	vec4 matlInfo   = texelFetch( matlInfoTex, matlIdx+1 );  // Get diffuse color
	vec4 matlTexIds = texelFetch( matlInfoTex, matlIdx+3 );  // Get matl texture IDs
	
	// Look up this particular fragments' diffuse texture color
	vec4 difTexColor = vec4(1.0);
	if (matlTexIds.y >= 0.0)
		difTexColor = texture( matlTextures, vec3( fragTexCoord.xy, matlTexIds.y ) );

	// Find:  Where is the light?
	vec4 esLightPos = view * vec4( wslightPos, 1.0 );

	// Compute important vectors
    vec3 esNorm  =  normalize(esNormal.xyz);
	vec3 toEye   = -normalize(esPosition.xyz);
	vec3 toLight =  normalize(esLightPos.xyz-esPosition.xyz);
	
	// Compute a simple diffuse shading
	float NdotL  =  clamp(dot(toLight,esNorm),0,1);
	result = vec4( NdotL * lightIntensity * lightColor * matlInfo.xyz * difTexColor.xyz, 1.0 );
}