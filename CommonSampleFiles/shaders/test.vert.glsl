#version 430 

// Tells IGLU to enable depth testing when using this shader
#pragma IGLU_ENABLE  DEPTH_TEST

layout(location = IGLU_VERTEX)   in vec3 vertex;   // Sends vertex data from models here
layout(location = IGLU_NORMAL)   in vec3 normal;   // Sends normal data (if any) from models here
layout(location = IGLU_TEXCOORD) in vec2 texCoord; // Sends texture coordinate data (if any) from models here

uniform mat4 project;     // The projection/perspective matrix
uniform mat4 modelview;   // The modelview matrix
uniform samplerBuffer InstanceData;
layout ( binding = 0) buffer BufferObject
{
    int faceIds[];
};
out vec4 fragNormal;
out float flag;
// This is the actual code that is executed.
void main( void )
{  
	int id = faceIds[2048];
	

  vec4 z = texelFetch(InstanceData,id*2);
  if (abs(z.x-8*id)<0.01)
      flag = 1;
  else
	  flag = 0;
  gl_Position = project * ( modelview * vec4( vertex, 1.0f ) ); 
  fragNormal = inverse( transpose( modelview ) ) * vec4( normal, 0.0 );
}