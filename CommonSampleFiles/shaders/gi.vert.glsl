
#version 430 core

layout(location = IGLU_VERTEX)   in vec3 vertex;   // Sends vertex data from models here
layout(location = IGLU_NORMAL)   in vec3 normal;   // Sends normal data (if any) from models here
layout(location = IGLU_TEXCOORD) in vec2 texCoord; // Sends texture coordinate data (if any) from models here
layout(location = IGLU_MATL_ID)  in float matlID;  // Sends ID for current material from model (if any)

uniform mat4 project;             // The projection/perspective matrix
uniform mat4 model;               // The model matrix
uniform mat4 view;                // The view matrix
uniform mat4 mirrorModel;  //The model matrix of mirror
uniform samplerBuffer InstanceData;

layout ( binding = 0) buffer BufferObject
{
    int faceIds[];
};
out vec4 esNormal;
out vec4 esPosition;
out vec2 fragTexCoord;
out float fragMatlID;
out vec3 esMirror;
out vec3 esMirrorNormal;
flat out int instanceId;
out float discardFlag;//顶点在镜子背面
out float flag;
//计算反射虚顶点，xyz是位置，w是discard标志
vec4 Reflection(vec3 fPoint, vec3 fNormal, vec3 inPoint )
{	
	float dir = dot(inPoint-fPoint,fNormal);
	vec4 result;
	if (dir < 0)
		result.w = -1.0;
	else
	{		
		result.w = 1.0;		
	}
	result.xyz = inPoint-2.0*dir*fNormal;
	return result;
}


void main( void )
{ 
	mat4 mv = view * model;
	vec4 eyePos = mv * vec4( vertex, 1.0f );
	esPosition = eyePos;
	//gl_Position = project * ( eyePos ); 
	esNormal = inverse( transpose( mv ) ) * vec4( normal, 0.0 );
	fragMatlID   = matlID;
	fragTexCoord = texCoord;

	

	//int id = gl_InstanceID/**2*/;

	int id = faceIds[gl_InstanceID];
	
	instanceId = id;
	mat4 mmv = view * mirrorModel;
	vec4 fcenter = mmv* texelFetch(InstanceData, id*2);
	vec4 fnormal= texelFetch(InstanceData, id*2+1);
	/*vec4 fcenter =   view*vec4(5.0,0,0,1);
	vec4 fnormal= vec4(-1,0,0,0);*/
	
	/*fnormal.xyz=normalize(mat3(mv)*fnormal.xyz);*/
	//vec4 fnormal = vec4(1,0,0,0);inverse( transpose( view * model ) ) 
	fnormal = inverse(transpose(mmv)) * fnormal;
	esMirrorNormal = normalize(fnormal.xyz);
	
	/*vec3 virtualPos=Reflection(fcenter.xyz/fcenter.w, fnormal.xyz, eyePos.xyz/eyePos.w);		*/
	esMirror= fcenter.xyz/fcenter.w;
	vec3 inpoint = eyePos.xyz/eyePos.w;
	//fnormal.xyz=normalize(mat3(mv)*fnormal.xyz);
	vec4 virtualPos;
	/*vec4 fakeCenter = vec4(5.0,0,0,1);
	vec4 fakeNormal = vec4(-1,0,0,0);
	fakeCenter = (mv*fakeCenter);
	fakeNormal = inverse(transpose(mv))*fakeNormal;
	if (false)	
		virtualPos=Reflection(fakeCenter.xyz, fakeNormal.xyz, inpoint.xyz);
	else*/
	//virtualPos = Reflection(fcenter.xyz, fnormal.xyz, inpoint.xyz);

	float dir = dot(inpoint.xyz-esMirror,esMirrorNormal);
	virtualPos.xyz = inpoint.xyz-2.0*dir*esMirrorNormal;
	discardFlag = dir;
	gl_Position = project * vec4(virtualPos.xyz,1.0);
		
   
}