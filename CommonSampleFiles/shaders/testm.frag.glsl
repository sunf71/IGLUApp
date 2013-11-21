#version 430 core 
flat in float index;
uniform sampler2D   stencilTexture;
out vec4 result;


void main()
{
	vec2 fragCorrd = vec2((gl_FragCoord.x)/512,(gl_FragCoord.y)/512);
	
	//float dir = dot(esPosition-esMirror,esMirrorNormal);	
	
	//ÃÉ°æ²âÊÔ
	vec3 texvalue = texture( stencilTexture, fragCorrd).xyz;
	float id= (texvalue.x+texvalue.y+texvalue.z) - 1.0;	
	
	if(abs(id-index)>0.00001)
	{	
		result = vec4(0,abs(id-index),0,1);
	}
	else
	{
	if (abs(index - 1) < 0.001)
	   result =   vec4(0,   1.0 ,   0.0, 1.0  );
	else
		result = vec4(1,   0.0 ,   0.0, 1.0  );
		if ( abs(index+1.0 - 2047) < 0.001)
		result = vec4(1,0,0,1);
	if ( abs(index+1.0 - 2048) < 0.001)
		result = vec4(0,1,0,1);
	if ( abs(index+1.0 - 2049) < 0.001)
		result = vec4(0,0,1,1);
	if ( abs(index+1.0 - 2050) < 0.001)
		result = vec4(1,1,1,1);
	}
	
}