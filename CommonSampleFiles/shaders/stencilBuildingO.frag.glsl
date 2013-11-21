#version 430 core 
flat in float indexToFrag;
out vec4 result;


void main()
{
	result =vec4(indexToFrag+1.0,   0.0 ,   0.0, 1.0  );
}