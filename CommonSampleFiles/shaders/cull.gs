#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 OrigPosition[1];
flat in int objectVisible[1];

out vec4 CulledPosition;

void main() {

   /* only emit primitive if the object is visible */
   if ( objectVisible[0] == 1 )
   {
      CulledPosition = OrigPosition[0];
      EmitVertex();
      EndPrimitive();
   }
}
