#version 430 

in vec3 vertex;
uniform mat4 mvp;

// This is the actual code that is executed.
void main( void )
{  
	gl_Position = mvp*vec4(vertex,1.0);
}