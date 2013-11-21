#version 330 core

uniform 	mat4 ModelViewMatrix;
uniform	mat4 ProjectionMatrix;
uniform 	float TimeFactor;


uniform vec3 SkyLightDir;
uniform vec3 FogColor;

//uniform samplerBuffer InstanceData;

layout(location = IGLU_VERTEX)   in vec3 VertexPosition;   // Sends vertex data from models here
layout(location = IGLU_NORMAL)   in vec3 VertexNormal;   // Sends normal data (if any) from models here
layout(location = IGLU_TEXCOORD) in vec2 VertexTexCoord; // Sends texture coordinate data (if any) from models here
layout(location = IGLU_MATL_ID)  in float matlID;  // Sends ID for current material from model (if any)
layout(location = 5)   in vec4 InstancePosition;  // instance position
layout(location = 6)   in vec4 InstanceNormal;  // instance normal
//in vec3 VertexPosition;
//in vec3 VertexNormal;
//in vec2 VertexTexCoord;
//in vec4 InstancePosition;
//const float DeformRatio = 1.0;

out vec4 FogParam;
out vec3 InterpNormal;
out vec2 InterpTexCoord;
out vec3 InterpViewDir;
out vec3 InterpLightDir;
out float fragMatlID;
out vec4 InstanceData;
void main(void) {

	/* fetch instance position from texture buffer and add the local vertex coordinate to it */
	vec4 ObjectSpacePosition = InstancePosition + vec4(VertexPosition, 1.0);
	//vec4 ObjectSpacePosition = vec4(VertexPosition, 1.0) + InstancePosition;
	/* hackish deformation to produce wind-like effect */
	float DeformRatio = 1.0 / (0.7 - ( VertexPosition.y )) / 50.0;
	ObjectSpacePosition.x += DeformRatio * (cos( TimeFactor )+1.0) * 2.0;
	ObjectSpacePosition.y += DeformRatio * (sin( TimeFactor )+1.0) * 0.5;
	
	/* transform into view space */
	vec4 ViewSpacePosition = ModelViewMatrix * ObjectSpacePosition;
	
	/* calculate fog related information */
	FogParam.xyz = FogColor;
	FogParam.w = exp(length(ViewSpacePosition.xyz) / 50.0) / 2.6;
    
	/* transform into clip space */
	gl_Position = ProjectionMatrix * ViewSpacePosition;
	
	/* pass through other attributes */
	InterpNormal = VertexNormal;
	InterpTexCoord = VertexTexCoord;	
	InterpViewDir = vec3( 0.0, 0.0, 1.0 );
	InterpLightDir = SkyLightDir; 
	fragMatlID = matlID;
	InstanceData = InstancePosition;
}
