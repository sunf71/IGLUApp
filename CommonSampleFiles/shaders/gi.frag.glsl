#version 400

uniform mat4 view; 
uniform float lightIntensity;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform samplerBuffer  matlInfoTex;
uniform sampler2DArray matlTextures;
uniform float resX;
uniform float resY;
in vec4  esNormal;
in vec4  esPosition;
in vec2  fragTexCoord;
in float fragMatlID;
in float flag;

uniform sampler2D   stencilTexture;
flat in int instanceId;
in vec3 esMirror;
in vec3 esMirrorNormal;
in float discardFlag;

out vec4 result;

void main( void )
{
	vec2 fragCorrd = vec2((gl_FragCoord.x)/resX,(gl_FragCoord.y)/resY);
	
	//float dir = dot(esPosition-esMirror,esMirrorNormal);	
	vec4 texvalue = texture( stencilTexture, fragCorrd);
	float id= (texvalue.x+texvalue.y+texvalue.z) - 1.0;	
	//蒙版测试
	//float id=texture( stencilTexture, fragCorrd).x- 1.0;	

	if(abs(id-instanceId)>0.01)
	{	
		discard;
	
	}

	////镜面在顶点前面
	if (discardFlag < 0)
		discard;

	// Look up data about this particular fragment's material
	vec4 matlInfo = vec4(1.0);
	vec4 matlTexIds = vec4(-1.0);
	int matlIdx     = 4 * int(fragMatlID+0.5);
	if (matlIdx > 0)
	{
		matlInfo   = texelFetch( matlInfoTex, matlIdx+1 );  // Get diffuse color
		matlTexIds = texelFetch( matlInfoTex, matlIdx+3 );  // Get matl texture IDs

	}
	
	// Look up this particular fragments' diffuse texture color
	vec4 difTexColor = vec4(1.0);
	if (matlTexIds.y >= 0.0)
		difTexColor = texture( matlTextures, vec3( fragTexCoord.xy, matlTexIds.y ) );

	// Find:  Where is the light?
	vec4 esLightPos = view * vec4( lightPos, 1.0 );

	// Compute important vectors
    vec3 esNorm  =  normalize(esNormal.xyz);
	vec3 toEye   = -normalize(esPosition.xyz);
	vec3 toLight =  normalize(esLightPos.xyz-esPosition.xyz);
	
	// Compute a simple diffuse shading
	float NdotL  =  clamp(dot(toEye,esNorm),0,1)+1;
	result = vec4( NdotL * lightIntensity * lightColor * matlInfo.xyz * difTexColor.xyz, 1.0 );

}