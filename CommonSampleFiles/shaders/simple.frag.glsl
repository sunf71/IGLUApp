#version 430 

out vec4 frag;
uniform vec4 vcolor;

void main(void)
{
	//vec4 color = vec4(1.0,0,0,1.0);
	frag = vcolor;
}