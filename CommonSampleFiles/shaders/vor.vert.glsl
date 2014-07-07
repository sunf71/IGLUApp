
#version 430 core

layout(location = IGLU_VERTEX)   in vec3 vertex;   // Sends vertex data from models here
layout(location = IGLU_NORMAL)   in vec3 normal;   // Sends normal data (if any) from models here
layout(location = IGLU_TEXCOORD) in vec2 texCoord; // Sends texture coordinate data (if any) from models here
layout(location = IGLU_MATL_ID)  in float matlID;  // Sends ID for current material from model (if any)
layout(location = 10) in float mirrorId;
uniform mat4 project;             // The projection/perspective matrix
uniform mat4 model;               // The model matrix
uniform mat4 view;                // The view matrix
uniform mat4 mirrorModel;  //The model matrix of mirror
uniform samplerBuffer InstanceData;


out vec4 esNormal;
out vec4 esPosition;
out vec2 fragTexCoord;
out float fragMatlID;
out vec3 esMirror;
out vec3 esMirrorNormal;
flat out int instanceId;

//计算反射虚顶点，xyz是位置，w是discard标志
vec3 Reflection(vec3 fPoint, vec3 fNormal, vec3 inPoint )
{	
	float dir = dot(inPoint-fPoint,fNormal);
	vec3 result = inPoint-2.0*dir*fNormal;
	return result;
}


void main( void )
{ 
	mat4 mv = view * model;
	vec4 eyePos = mv * vec4( vertex, 1.0f );
	esPosition = eyePos;
	
	esNormal = inverse( transpose( mv ) ) * vec4( normal, 0.0 );
	fragMatlID   = matlID;
	fragTexCoord = texCoord;


	
	mat4 mmv = view * mirrorModel;
	vec4 fcenter = mmv* texelFetch(InstanceData, 2);
	vec4 fnormal= texelFetch(InstanceData, 3);
	instanceId = int(mirrorId+0.5);
	fcenter = mmv* texelFetch(InstanceData, instanceId*2);
	fnormal =  texelFetch(InstanceData,instanceId*2+1);
	//if (mirrorId == 0)
	//{
	//	fcenter = mmv* texelFetch(InstanceData, instanceId*2);
	//	fnormal =  texelFetch(InstanceData,instanceId*2+1);
	//}
	//else
	//{
	//	fcenter = mmv* texelFetch(InstanceData, 2);
	//	fnormal =  texelFetch(InstanceData,3);
	//}
	fnormal = inverse(transpose(mmv)) * fnormal;
	esMirrorNormal = normalize(fnormal.xyz);
	
	
	esMirror= fcenter.xyz/fcenter.w;
	vec3 inpoint = eyePos.xyz/eyePos.w;
	
	vec3 virtualPos = Reflection(esMirror,esMirrorNormal,inpoint);	
	
	gl_Position = project*vec4(virtualPos,1.0);
}