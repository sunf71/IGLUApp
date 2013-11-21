#version 400

in vec4 fragNormal;  // Input: per-fragment normal
out vec4 result;     // Output: 4-component color
in float flag;
// Display as grayscale the dot product of surface normal & view direction
void main( void )
{
  //result = vec4( abs(fragNormal.z),0,0, abs(fragNormal.z) );
  result = vec4( abs(fragNormal.z),flag,0, 1 );
}