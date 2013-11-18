#version 430 core
//用于绘制镜面蒙版
layout(location = 0)   in vec3 vertex;   // Sends vertex data from models here
layout(location = 1)   in vec3 normal;  
layout(location = 2)   in float vertexIndex;   // Sends normal data (if any) from models here
uniform mat4 project;     // The projection/perspective matrix
uniform mat4 model;   // The model and view matrix
uniform mat4 view;
out float index;
out vec4 esNormal;
out vec4 esPosition;
void main(){
	mat4 mv = view * model;
	esPosition = mv * vec4(vertex,1.0f);
    gl_Position = vec4(vertex,1.f);
	//gl_Position = project* esPosition;
    index = vertexIndex;
	esNormal = inverse(transpose(mv))*vec4(normal,0);	
}
