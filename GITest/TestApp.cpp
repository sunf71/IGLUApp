#include "TestApp.h"

TestApp::~TestApp()
{
	
}
void TestApp::InitBuffer()
{
	uint TriNum = 4900;
	float* tmp = new float[TriNum*8];
	for(int i=0; i<TriNum*8; i++)
		tmp[i] = i;
	_mirrorDataBuffer = new IGLUBuffer();

	_mirrorDataBuffer->SetBufferData(TriNum*8*sizeof(float),tmp);
	_mirrorBuffer = new IGLUTextureBuffer();
	_mirrorBuffer->BindBuffer(GL_RGBA32F,_mirrorDataBuffer);
	
	delete[] tmp;

	int * ssboData = (int*)malloc(sizeof(int)*TriNum);
	for(int i=0; i< TriNum; i++)
	{
		ssboData[i] = i;
	}
	glGenBuffers(1,&_SSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,_SSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,sizeof(int)*TriNum,ssboData,GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,_SSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,0);

	free(ssboData);
}
void TestApp::InitShaders()
{
	_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/test.vert.glsl","../../CommonSampleFiles/shaders/test.frag.glsl");
}

void TestApp::InitScene()
{
		for (int i=0; i<_sceneData->getObjects().size(); i++)
	{
		SceneObject* obj = _sceneData->getObjects()[i];
		switch (obj->getType())
		{
		case SceneObjType::mesh:
			{
				ObjModelObject* mesh = (ObjModelObject*)obj;
				//GLMmodel* model = glmReadOBJ(mesh->getObjFileName().c_str());
				IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str());
				//IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( model,IGLU_OBJ_COMPACT_STORAGE);
				_objReaders.push_back(objReader);
				glm::mat4 trans = (mesh->getTransform());					
				_objTransforms.push_back(IGLUMatrix4x4(&trans[0][0]));
				
				break;
			}
		default:
			break;
		}
	}
	// We've loaded all our materials, so prepare to use them in rendering
	IGLUOBJMaterialReader::FinalizeMaterialsForRendering(IGLU_TEXTURE_REPEAT);

	InitBuffer();

}

void TestApp::Display()
{	
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	_frameRate->StartFrame();

	 _objShader->Enable();
	_objShader["project"]         = _camera->GetProjectionMatrix();
	//_objShader["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
	//_objShader["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
	_objShader["InstanceData"] = _mirrorBuffer;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,_SSBO);	
	for(int i=0; i<_objReaders.size(); i++)
	{
		_objShader["modelview"]           = _camera->GetViewMatrix() * _objTransforms[i];			
		_objReaders[i]->Draw(_objShader);			

	}
	_objShader->Disable();
	//IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], IGLU_DRAW_FLIP_Y );
	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}