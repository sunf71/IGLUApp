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


out vec4 result;

void main( void )
{
	if(instanceId==7)
		result = vec4(0,0,1,1.0);
	else
		result = vec4(1,0,0,1.0);
}