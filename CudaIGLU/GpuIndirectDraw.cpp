#include "GpuIndirectDrawApp.h"
#include "cuda_runtime_api.h"

namespace OGL
{
	float random(float min,float max)
	{
		IGLURandom::Ptr rand = new IGLURandom();
		float ret =  rand->fRandom()*(max-min) + min;
		delete rand;
		return ret;

	}
	void GpuIndirectDrawApp::InitCudaOGL()
	{
		cudaDeviceProp prop;
		int dev;

		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 1;
		prop.minor = 0;
		cudaChooseDevice(&dev, &prop);
		cudaGLSetGLDevice(dev);
		_indirectDrawBuffer = new IGLUBuffer(IGLU_DRAW_INDIRECT);		
		_indirectDrawBuffer->SetBufferData(NUM_DRAWS * sizeof(DrawElementsIndirectCommand),
			NULL,IGLU_DYNAMIC);
		_indirectDrawBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&resource, _indirectDrawBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);

		
		InitAttribute();

	}
	void GpuIndirectDrawApp::UpdateDrawCommand()
	{
		uint primCount = _objReaders[0]->GetTriangleCount()*3;
		uint* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
		update(devPtr,primCount,NUM_DRAWS);
		cudaGraphicsUnmapResources(1, &resource, NULL);
	}


	void GpuIndirectDrawApp::InitScene()
	{
		IGLUApp::InitScene();

		InitInstanceData();		
	}
	void GpuIndirectDrawApp::InitInstanceData()
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
		//InitDrawElementCommand();
		InitCudaOGL();
	}
	void GpuIndirectDrawApp::InitDrawElementCommand()
	{
		_indirectDrawBuffer = new IGLUBuffer(IGLU_DRAW_INDIRECT);
		_indirectDrawBuffer->Bind();
		_indirectDrawBuffer->SetBufferData(NUM_DRAWS * sizeof(DrawElementsIndirectCommand),
			NULL);
		DrawElementsIndirectCommand * cmd = (DrawElementsIndirectCommand *)_indirectDrawBuffer->Map(IGLU_WRITE);	
				
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

		_indirectDrawBuffer->Unmap();
		_indirectDrawBuffer->Bind();

		glBindVertexArray(_objReaders[0]->GetVertexArray()->GetArrayID());

		_drawIndexBuffer = new IGLUBuffer();	
		unsigned* draw_index = new unsigned[NUM_DRAWS];
		for (unsigned i = 0; i < NUM_DRAWS; i++)
		{
			draw_index[i] = i;
		}
		_drawIndexBuffer->SetBufferData(NUM_DRAWS * sizeof(GLuint),draw_index,IGLU_STATIC|IGLU_DRAW);
		delete[] draw_index;

		_drawIndexBuffer->Bind();
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
	void GpuIndirectDrawApp::InitDrawArrayCommand()
	{
		_indirectDrawBuffer = new IGLUBuffer(IGLU_DRAW_INDIRECT);
		_indirectDrawBuffer->Bind();
		_indirectDrawBuffer->SetBufferData(NUM_DRAWS * sizeof(DrawArraysIndirectCommand),
			NULL);
		DrawArraysIndirectCommand * cmd = (DrawArraysIndirectCommand *)_indirectDrawBuffer->Map(IGLU_WRITE);	
		GLuint count = _objReaders[0]->GetTriangleCount()*3;
		for (unsigned i = 0; i < NUM_DRAWS; i++)
		{
			cmd[i].first=0;
			cmd[i].count = count;
			cmd[i].primCount = 1;
			cmd[i].baseInstance = i;
		}
		_indirectDrawBuffer->Unmap();
		_indirectDrawBuffer->Bind();

		glBindVertexArray(_objReaders[0]->GetVertexArray()->GetArrayID());

		_drawIndexBuffer = new IGLUBuffer();	
		unsigned* draw_index = new unsigned[NUM_DRAWS];
		for (unsigned i = 0; i < NUM_DRAWS; i++)
		{
			draw_index[i] = i;
		}
		_drawIndexBuffer->SetBufferData(NUM_DRAWS * sizeof(GLuint),draw_index,IGLU_STATIC|IGLU_DRAW);
		delete[] draw_index;

		_drawIndexBuffer->Bind();
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
	void GpuIndirectDrawApp::InitAttribute()
	{
		glBindVertexArray(_objReaders[0]->GetVertexArray()->GetArrayID());
		_drawIndexBuffer = new IGLUBuffer();	
		unsigned* draw_index = new unsigned[NUM_DRAWS];
		for (unsigned i = 0; i < NUM_DRAWS; i++)
		{
			draw_index[i] = i;
		}
		_drawIndexBuffer->SetBufferData(NUM_DRAWS * sizeof(GLuint),draw_index,IGLU_STATIC|IGLU_DRAW);
		delete[] draw_index;
		
		


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


		_drawIndexBuffer->Bind();
		
		glVertexAttribIPointer(10, 1, GL_UNSIGNED_INT, 0, NULL);
		glVertexAttribDivisor(10, 1);
		glEnableVertexAttribArray(10);
	}
	void GpuIndirectDrawApp::InitShaders()
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


	void GpuIndirectDrawApp::Display()
	{
		/*IGLUApp::Display();
		return;*/
		// Start timing this frame draw
		_frameRate->StartFrame();

		UpdateDrawCommand();
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
}