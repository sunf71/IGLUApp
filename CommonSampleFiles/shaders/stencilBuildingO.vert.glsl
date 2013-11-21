#version 430 core
//用于绘制镜面蒙版
layout(location = 0)   in vec3 vertex;   // Sends vertex data from models here
layout(location = 1)   in vec3 normal;  // Sends normal data (if any) from models here
layout(location = 2)   in float vertexIndex;   
uniform mat4 project;     // The projection/perspective matrix
uniform mat4 model;   // The model and view matrix
uniform mat4 view;
flat out float indexToFrag;

void main(){

	gl_Position = project* view * model * vec4(vertex,1.0f);
    indexToFrag = vertexIndex;
	
}
