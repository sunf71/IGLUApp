#version 430 core 
flat in float indexToFrag;
out vec4 result;


void main()
{
	float value = indexToFrag+1.0;
	if (value > 4094)
	{
		result = vec4(2047,2047,value-4094,1.0);
	}
	else if( value > 2047)
	{
		result = vec4(2047,value-2047,0,1.0);
	}
	else
	result =vec4(value,   0.0 ,   0.0, 1.0  );
	
}