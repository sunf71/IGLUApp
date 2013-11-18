

#version 430 core

// This really shouldn't be required, but NVIDIA's driver seems to want it...
#extension GL_EXT_geometry_shader4 : enable

// This says our geometry shader takes triangle primitives as input
//    and runs once for each input triangle (invocations=1 is the default)
layout (triangles, invocations = 1) in;

// This says our geometry shader outputs triangle-strip primitives and 
//    generates at most 3 vertices per geometry shader invocation.  The
//    only valid outputs are points, line_strip, triangle_strip.  The 
//    (pretty stupid) default for max_vertices is 0.
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 project;
uniform mat4 model;   // The modelview matrix
uniform mat4 view;
in float index[];
in vec4 esNormal[];
in vec4 esPosition[];
flat out float indexToFrag;
layout (binding = 0, offset = 0) uniform atomic_uint faces;
layout ( binding = 0) buffer BufferObject
{
    int faceIds[];
};
void main( void )
{
	vec4 center = esPosition[0];
	vec3 toEye   = -normalize(center.xyz);
	vec3 normal = normalize(esNormal[0].xyz);
	mat4 mvp = project *  view * model; 
	//cull  back face
	if (dot(toEye,normal) > 0)
	{
		int i;
		for(i = 0;i < gl_in.length();i++)
		{
			indexToFrag = index[i];
			gl_Position = mvp*gl_in[i].gl_Position;
			EmitVertex();
		}
		//记录有效面个数
		uint counter = atomicCounterIncrement(faces);
		faceIds[counter] = int(index[0]);
		
		EndPrimitive();
	}
}