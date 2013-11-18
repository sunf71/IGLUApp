#version 400 
//用于贴图，将GI绘制结果作为纹理贴到镜面上
layout(location = 0)   in vec3 vertex;   // Sends vertex data from models here
layout(location = 1)   in vec3 normal;   // Sends normal data (if any) from models here
layout(location = 2)   in float vertexIndex;   // Sends normal data (if any) from models here
uniform mat4 project;     // The projection/perspective matrix
uniform mat4 modelview;   // The modelview matrix

out vec4 esNormal;
out vec4 esPosition;
out vec2 fragTexCoord;
out float fragMatlID;

void main(){

    vec4 eyePos = modelview * vec4( vertex, 1.0f );
	esPosition = eyePos;
	gl_Position = project * ( eyePos ); 
	esNormal = inverse( transpose( modelview ) ) * vec4( normal, 0.0 );
	
	fragTexCoord = gl_Position.xy/gl_Position.w*0.5+0.5;
   
}
