#version 330 core


uniform samplerBuffer  matlInfoTex;
uniform sampler2DArray matlTextures;
in vec4 FogParam;
in vec3 InterpNormal;
in vec2 InterpTexCoord;
in vec3 InterpViewDir;
in vec3 InterpLightDir;
in float fragMatlID;
in vec4 InstanceData;
out vec4 FragmentColor;

void main(void)
{
    // Look up data about this particular fragment's material
	int matlIdx     = 4 * int(fragMatlID+0.5);
	vec4 matlInfo   = texelFetch( matlInfoTex, matlIdx+1 );  // Get diffuse color
	vec4 matlTexIds = texelFetch( matlInfoTex, matlIdx+3 );  // Get matl texture IDs
    // Look up this particular fragments' diffuse texture color
	vec4 color = vec4(1.0);
	if (matlTexIds.y >= 0.0)
		color = texture( matlTextures, vec3( InterpTexCoord.xy, matlTexIds.y ) );
    
    
    /* discard fragments with small alpha (alpha testing for leaf texture) */
    if (color.a < 0.5) discard;

    /* normalize and calculate vectors needed for light computation */
    vec3 Normal = normalize(InterpNormal);
    vec3 ViewDir = normalize(InterpViewDir);
    vec3 LightDir = normalize(InterpLightDir);
    vec3 Reflection = reflect(LightDir, Normal);

	/* calculate basic phong lighting */
	float Diffuse = clamp(dot(Normal, LightDir), 0.0, 1.0);
	float Specular = pow(clamp(dot(Reflection, ViewDir), 0.0, 1.0), 4);
	
   
	/* accumulate final fragment color */    
  	FragmentColor = mix( color * ( 0.3 + Diffuse + Specular ), FogParam, FogParam.w );
   

}
