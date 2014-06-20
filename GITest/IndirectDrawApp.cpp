#include "IndirectDrawApp.h"
float random(float min,float max)
{
	IGLURandom::Ptr rand = new IGLURandom();
	float ret =  rand->fRandom()*(max-min) + min;
	delete rand;
	return ret;

}
void IndirectDrawApp::InitScene()
{
	IGLUApp::InitScene();

	InitInstanceData();		
}
void IndirectDrawApp::InitInstanceData()
{
	_instanceData = new InstanceData[NUM_DRAWS];		

	int size = (int)sqrt((float)NUM_DRAWS);
	for (int i=0; i<size; i++) {
		for (int j=0; j<size; j++) {
			_instanceData[i+j*size].position = vec4( (i-size/2)*4.0f+random(-0.05,0.05), -5.5f, (j-size/2)*4.0f+random(-0.05,0.05), 0.0f );
			_instanceData[i+j*size].normal = vec4(1,0,0,0);

		}
	}

	_grassBuffer = new IGLUBuffer();			
	_grassBuffer->SetBufferData(sizeof(InstanceData)*NUM_DRAWS,_instanceData);			
	_grassTexBuffer = new IGLUTextureBuffer();
	_grassTexBuffer->BindBuffer(GL_RGBA32F,_grassBuffer);

	//InitDrawArrayCommand();
	InitDrawElementCommand();
}
void IndirectDrawApp::InitDrawElementCommand()
{
	glGenBuffers(1, &indirect_draw_buffer);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_draw_buffer);
	

	DrawElementsIndirectCommand * cmd = new DrawElementsIndirectCommand[NUM_DRAWS];
	GLuint count = _objReaders[0]->GetTriangleCount()*3;
	GLuint vcount = _objReaders[0]->GetVertecies().size();
	for (unsigned i = 0; i < NUM_DRAWS; i++)
	{
		cmd[i].firstIndex=0;
		cmd[i].count = count;
		cmd[i].instanceCount = 1;
		cmd[i].baseInstance = i;
		cmd[i].baseVertex = 0;
	}

	glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawElementsIndirectCommand) * NUM_DRAWS, cmd, GL_STATIC_DRAW);

	delete[] cmd;

	glBindVertexArray(_objReaders[0]->GetVertexArray()->GetArrayID());

	glGenBuffers(1, &draw_index_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, draw_index_buffer);
	glBufferData(GL_ARRAY_BUFFER,
		NUM_DRAWS * sizeof(GLuint),
		NULL,
		GL_STATIC_DRAW);

	GLuint * draw_index =
		(GLuint *)glMapBufferRange(GL_ARRAY_BUFFER,
		0,
		NUM_DRAWS * sizeof(GLuint),
		GL_MAP_WRITE_BIT |
		GL_MAP_INVALIDATE_BUFFER_BIT);

	for (unsigned i = 0; i < NUM_DRAWS; i++)
	{
		draw_index[i] = i;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribIPointer(10, 1, GL_UNSIGNED_INT, 0, NULL);
	glVertexAttribDivisor(10, 1);
	glEnableVertexAttribArray(10);


	_objReaders[0]->GetVertexArray()->GetVertexBuffer()->Bind();

	//   We'll have 1 float for a material ID
	//   We'll have 1 float for an object ID
	//   We'll have 3 floats (x,y,z) for each of the 3 verts of each triangle 
	//   We'll have 3 floats (x,y,z) for each of the 3 normals of each triangle
	//   We'll have 2 floats (u,v) for each of the 3 texture coordinates of each triangle
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 3, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (8) );
	glEnableVertexAttribArray( 1 );
	glVertexAttribPointer( 1, 3, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (20) );
	glEnableVertexAttribArray( 2 );
	glVertexAttribPointer( 2, 2, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (32) );
	glEnableVertexAttribArray( 3 );
	glVertexAttribPointer( 3, 1, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (0) );
	glEnableVertexAttribArray( 4 );
	glVertexAttribPointer( 4, 1, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (4) );

}
void IndirectDrawApp::InitDrawArrayCommand()
{
	glGenBuffers(1, &indirect_draw_buffer);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_draw_buffer);
	glBufferData(GL_DRAW_INDIRECT_BUFFER,
		NUM_DRAWS * sizeof(DrawArraysIndirectCommand),
		NULL,
		GL_STATIC_DRAW);

	DrawArraysIndirectCommand * cmd = (DrawArraysIndirectCommand *)
		glMapBufferRange(GL_DRAW_INDIRECT_BUFFER,
		0,
		NUM_DRAWS * sizeof(DrawArraysIndirectCommand),
		GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	GLuint count = _objReaders[0]->GetTriangleCount()*3;
	for (unsigned i = 0; i < NUM_DRAWS; i++)
	{
		cmd[i].first=0;
		cmd[i].count = count;
		cmd[i].primCount = 1;
		cmd[i].baseInstance = i;
	}

	glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER);


	glBindVertexArray(_objReaders[0]->GetVertexArray()->GetArrayID());

	glGenBuffers(1, &draw_index_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, draw_index_buffer);
	glBufferData(GL_ARRAY_BUFFER,
		NUM_DRAWS * sizeof(GLuint),
		NULL,
		GL_STATIC_DRAW);

	GLuint * draw_index =
		(GLuint *)glMapBufferRange(GL_ARRAY_BUFFER,
		0,
		NUM_DRAWS * sizeof(GLuint),
		GL_MAP_WRITE_BIT |
		GL_MAP_INVALIDATE_BUFFER_BIT);

	for (unsigned i = 0; i < NUM_DRAWS; i++)
	{
		draw_index[i] = i;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribIPointer(10, 1, GL_UNSIGNED_INT, 0, NULL);
	glVertexAttribDivisor(10, 1);
	glEnableVertexAttribArray(10);


	_objReaders[0]->GetVertexArray()->GetVertexBuffer()->Bind();

	//   We'll have 1 float for a material ID
	//   We'll have 1 float for an object ID
	//   We'll have 3 floats (x,y,z) for each of the 3 verts of each triangle 
	//   We'll have 3 floats (x,y,z) for each of the 3 normals of each triangle
	//   We'll have 2 floats (u,v) for each of the 3 texture coordinates of each triangle
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 3, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (8) );
	glEnableVertexAttribArray( 1 );
	glVertexAttribPointer( 1, 3, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (20) );
	glEnableVertexAttribArray( 2 );
	glVertexAttribPointer( 2, 2, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (32) );
	glEnableVertexAttribArray( 3 );
	glVertexAttribPointer( 3, 1, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (0) );
	glEnableVertexAttribArray( 4 );
	glVertexAttribPointer( 4, 1, GL_FLOAT, false, 10*sizeof(float), (char *)NULL + (4) );

	
}

void IndirectDrawApp::InitShaders()
{
	IGLUShaderProgram::Ptr dftShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl",
		"../../CommonSampleFiles/shaders/object.frag.glsl");
	_shaders.push_back(dftShader);

	dftShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/Indirect.vert.glsl",
		"../../CommonSampleFiles/shaders/Indirect.frag.glsl");
	dftShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 
	//dftShader->SetProgramDisables( IGLU_GLSL_BLEND );
	_shaders.push_back(dftShader);
	
}


void IndirectDrawApp::Display()
{
	/*IGLUApp::Display();
	return;*/
	// Start timing this frame draw
	_frameRate->StartFrame();

	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	_dtTime += 0.02;		
	_shaders[1]->Enable();

	_shaders[1]["ModelViewMatrix"] = _camera->GetViewMatrix();;
	_shaders[1]["ProjectionMatrix"] = _camera->GetProjectionMatrix();
	_shaders[1]["TimeFactor"] = _dtTime;
	_shaders[1]["SkyLightDir"] = vec3(-0.316227766f, 0.948683298f, 0.0f);
	_shaders[1]["FogColor"] = vec3(0.f);
	_shaders[1]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
	_shaders[1]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
	_shaders[1]["InstanceData"] = _grassTexBuffer;
	
	IGLUVertexArray::Ptr vao = _objReaders[0]->GetVertexArray();
	vao->Bind();
	vao->GetElementArray()->Bind();
	//glMultiDrawArraysIndirect(GL_TRIANGLES, NULL, NUM_DRAWS, 0);
    glMultiDrawElementsIndirect(	GL_TRIANGLES, 	GL_UNSIGNED_INT, 	NULL, 	NUM_DRAWS, 	0);
	//glDrawArrays(GL_TRIANGLES, 0,3*_objReaders[0]->GetTriangleCount());
	//glDrawElements(GL_TRIANGLES,3*_objReaders[0]->GetTriangleCount(),GL_UNSIGNED_INT,0);
	vao->GetElementArray()->Unbind();
	vao->Unbind();
	//_objReaders[0]->Draw(_shaders[1]);
	_shaders[1]->Disable();
	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}