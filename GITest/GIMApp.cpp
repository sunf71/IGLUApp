#include "GIMApp.h"
#include<glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <time.h>




GIMApp::~GIMApp()
{	
	safe_delete(_mirrorBuffer);
	safe_delete(_mirrorVAO);
	safe_delete(_VAOBuffer);
	safe_delete(_cpuTimer);
	safe_delete(_gpuTimer);

	//�Ѿ������ݱ�����buffer��
	safe_delete( _mirrorDataBuffer);
	safe_delete( _mirrorBuffer);
	safe_delete( _mirrorUB);

	_instanceDatum.clear();
	
}
void GIMApp::InitBufferOld()
{
	IGLUOBJReader::Ptr _mirrorMesh = _objReaders[_mirrorId];
	//���澵��ķ��ߺ͵�
	int TriNum = _mirrorMesh->GetTriangleCount();	
	_instanceDatum = vector<InstanceData> (TriNum);
	uint *indexBuffer =(uint*) malloc(TriNum*3*sizeof(uint));

	//���ƾ����ɰ�;���ʱ��vao
	_mirrorVAO = new IGLUVertexArray();	

	//ÿ�������εĶ�����vertex 3,normal 3,index��7��������
	_VAOBuffer = (float*)malloc(TriNum*3*7*sizeof(float));

	std::vector<IGLUOBJTri *> tris = _mirrorMesh->GetTriangles();
	std::vector<vec3> vertices = _mirrorMesh->GetVertecies();
	std::vector<vec3> normals = _mirrorMesh->GetNormals();
	for (uint i=0; i<3*TriNum; i++)
		indexBuffer[i] = i;


	for(int i=0; i<TriNum; i++)
	{
		vec3 v[3];		
		for(int j=0; j<3; j++)
		{
			_instanceDatum[i].position[j] = vertices[tris[i]->vIdx[0]].GetElement(j);	
			v[j] = vertices[tris[i]->vIdx[j]];
			for(int k=0; k<3; k++)
			{
				_VAOBuffer[i*21+j*7+k] = vertices[tris[i]->vIdx[j]].GetElement(k);
			}
			for(int k=0; k<3; k++)
			{				
				_VAOBuffer[i*21+j*7+k+3] = normals[tris[i]->nIdx[j]].GetElement(k);
			}
			_VAOBuffer[i*21+j*7+6] = i;
		}	
		vec3 faceNormal = (v[2]-v[1]).Cross(v[0]-v[1]);
		faceNormal.Normalize();
		
		for(int j=0; j<3; j++)
		{		
			_instanceDatum[i].normal[j] = faceNormal.GetElement(j);
		}		
	}

	/*IGLUMatrix4x4 model = _objTransforms[_mirrorId];
	IGLUMatrix4x4 normalM = model.Transpose();
	glm::mat4 mat = glm::make_mat4(normalM.GetDataPtr());
	mat = glm::inverse(mat);
	normalM = IGLUMatrix4x4(&mat[0][0]);
	for( int i=0; i< TriNum; i++)
	{
		_instanceDatum[i].position = model * _instanceDatum[i].position;
		_instanceDatum[i].normal = vec4((normalM * _instanceDatum[i].normal).xyz(),0);

	}*/

	float* tmp = &_instanceDatum[0].position[0];
	
	/*_mirrorUB = new IGLUBuffer(IGLU_UNIFORM);
	_mirrorUB->SetBufferData(TriNum*2*4*sizeof(float),tmp);	*/
	_mirrorDataBuffer = new IGLUBuffer();
	_mirrorDataBuffer->SetBufferData(TriNum*2*4*sizeof(float),tmp);

	_mirrorVAO->SetVertexArray(TriNum*3*7*sizeof(float),_VAOBuffer);
	_mirrorVAO->SetElementArray(GL_UNSIGNED_INT,TriNum*3*sizeof(uint),indexBuffer);

	_mirrorVAO->EnableAttribute(0,3,GL_FLOAT,7*sizeof(float));
	_mirrorVAO->EnableAttribute(1,3,GL_FLOAT,7*sizeof(float),BUFFER_OFFSET(3*sizeof(float)));	
	_mirrorVAO->EnableAttribute(2,1,GL_FLOAT,7*sizeof(float),BUFFER_OFFSET(6*sizeof(float)));
	free(indexBuffer);

	_mirrorStencilFBO =  IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,false);
	_mirrorFBO = IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,true);

}
void GIMApp::InitBuffer()
{	
	InitBufferOld();
	return;
	IGLUOBJReader::Ptr _mirrorMesh = _objReaders[_mirrorId];
	//���澵��ķ��ߺ͵�
	int TriNum = _mirrorMesh->GetTriangleCount();	
	_instanceDatum = vector<InstanceData> (TriNum);
	uint *indexBuffer =(uint*) malloc(TriNum*3*sizeof(uint));

	//���ƾ����ɰ�;���ʱ��vao
	_mirrorVAO = new IGLUVertexArray();	

	//ÿ�������εĶ�����vertex 3,normal 3,index��7��������
	_VAOBuffer = (float*)malloc(TriNum*3*7*sizeof(float));

	std::vector<IGLUOBJTri *> tris = _mirrorMesh->GetTriangles();
	std::vector<vec3> vertices = _mirrorMesh->GetVertecies();
	std::vector<vec3> normals = _mirrorMesh->GetNormals();
	for (uint i=0; i<3*TriNum; i++)
		indexBuffer[i] = i;


	for(int i=0; i<TriNum; i++)
	{
		vec3 v[3];		
		for(int j=0; j<3; j++)
		{
			_instanceDatum[i].position[j] = vertices[tris[i]->vIdx[0]].GetElement(j);	
			v[j] = vertices[tris[i]->vIdx[j]];
			for(int k=0; k<3; k++)
			{
				_VAOBuffer[i*21+j*7+k] = vertices[tris[i]->vIdx[j]].GetElement(k);
			}
			for(int k=0; k<3; k++)
			{				
				_VAOBuffer[i*21+j*7+k+3] = normals[tris[i]->nIdx[j]].GetElement(k);
			}
			_VAOBuffer[i*21+j*7+6] = i;
		}	
		vec3 faceNormal = (v[2]-v[1]).Cross(v[0]-v[1]);
		faceNormal.Normalize();
		
		for(int j=0; j<3; j++)
		{		
			_instanceDatum[i].normal[j] = faceNormal.GetElement(j);
		}		
	}

	IGLUMatrix4x4 model = _objTransforms[_mirrorId];
	IGLUMatrix4x4 normalM = model.Transpose();
	glm::mat4 mat = glm::make_mat4(normalM.GetDataPtr());
	mat = glm::inverse(mat);
	normalM = IGLUMatrix4x4(&mat[0][0]);
	for( int i=0; i< TriNum; i++)
	{
		_instanceDatum[i].position = model * _instanceDatum[i].position;
		_instanceDatum[i].normal = vec4((normalM * _instanceDatum[i].normal).xyz(),0);

	}
	/*for( int i=0; i< TriNum; i++)
	{
	printf("instance\n");
	printf("v %f, %f, %f, %f\n n %f, %f, %f, %f\n",instances[i].position[0],instances[i].position[1],instances[i].position[2],instances[i].position[3],
	instances[i].normal[0],instances[i].normal[1],instances[i].normal[2],instances[i].normal[3]);
	}*/
	/*for( int i=0; i< TriNum*3*7; i+=7)
	{
	
	printf("v %f, %f, %f, n %f, %f, %f, index %f\n",VAOBuffer[i],VAOBuffer[i+1],VAOBuffer[i+2],VAOBuffer[i+3],VAOBuffer[i+4],
		VAOBuffer[i+5],VAOBuffer[i+6],VAOBuffer[i+7]);
	}*/
	float* tmp = &_instanceDatum[0].position[0];
	/*for( int i=0; i< TriNum*2*4; i+=8)
	{
	printf("tmpbuffer\n");
	printf("v %f, %f, %f, %f\n n %f, %f, %f, %f\n",tmp[i],tmp[i+1],tmp[i+2],tmp[i+3],
	tmp[i+4],tmp[i+5],tmp[i+6],tmp[i+7]);
	}*/
	_mirrorDataBuffer = new IGLUBuffer();
	_mirrorDataBuffer->SetBufferData(TriNum*2*4*sizeof(float),tmp);
	_mirrorBuffer = new IGLUTextureBuffer();
	_mirrorBuffer->BindBuffer(GL_RGBA32F,_mirrorDataBuffer);
	//float* p = (float*)_mirrorDataBuffer->Map();
	//for( int i=2047*2*4; i< TriNum*2*4; i+=8)
	//{
	////printf("tmpbuffer\n");
	//printf("%d : v %f, %f, %f, %f\n n %f, %f, %f, %f\n",i, p[i],p[i+1],p[i+2],p[i+3],
	//p[i+4],p[i+5],p[i+6],p[i+7]);
	//}
	//_mirrorDataBuffer->Unmap();

	_mirrorVAO->SetVertexArray(TriNum*3*7*sizeof(float),_VAOBuffer);
	_mirrorVAO->SetElementArray(GL_UNSIGNED_INT,TriNum*3*sizeof(uint),indexBuffer);

	_mirrorVAO->EnableAttribute(0,3,GL_FLOAT,7*sizeof(float));
	_mirrorVAO->EnableAttribute(1,3,GL_FLOAT,7*sizeof(float),BUFFER_OFFSET(3*sizeof(float)));	
	_mirrorVAO->EnableAttribute(2,1,GL_FLOAT,7*sizeof(float),BUFFER_OFFSET(6*sizeof(float)));


	free(indexBuffer);

	_mirrorStencilFBO =  IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,false);
	_mirrorFBO = IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,true);

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



	// Local variables
	GLuint *counters;
	// Generate a name for the buffer and create it by bind
	// the name to the generic GL_ATOMIC_COUNTER_BUFFER
	// binding point
	glGenBuffers(1, &_counterBuffer);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, _counterBuffer);
	// Allocate enough space for two GLuints in the buffer
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, 1 * sizeof(GLuint),
		NULL, GL_DYNAMIC_COPY);
	// Now map the buffer and initialize it
	counters = (GLuint*)glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,	GL_WRITE_ONLY);
	counters[0] = 0;
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	// Finally, bind the now initialized buffer to the 0th indexed
	// GL_ATOMIC_COUNTER_BUFFER binding point
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, _counterBuffer);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,0);
}
void GIMApp::InitScene()
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
				IGLUOBJReader::Ptr objReader;
				
				objReader = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(), IGLU_OBJ_UNITIZE);
		
				//IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( model,IGLU_OBJ_COMPACT_STORAGE);
				_objReaders.push_back(objReader);
				glm::mat4 trans = (mesh->getTransform());					
				_objTransforms.push_back(IGLUMatrix4x4(&trans[0][0]));
				if (!obj->getMaterialName().compare("mirror"))
				{
					_mirrorId = _objReaders.size()-1;		
					_mirrorTransform = _objTransforms[_objTransforms.size()-1];
				}			
				//delete model;
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

int GIMApp::UpdateMirrorVAO()
{
	float * mirrorVAO = new float[_instanceDatum.size()*3*7];
	vector<uint> indices;
	vector<InstanceData> instances;
	IGLUMatrix4x4 mv = _camera->GetViewMatrix();
	IGLUMatrix4x4 normalM = mv.Transpose();
	glm::mat4 mat = glm::make_mat4(normalM.GetDataPtr());
	mat = glm::inverse(mat);
	normalM = IGLUMatrix4x4(&mat[0][0]);
	for(int i=0,k=0; i<_instanceDatum.size(); i++)
	{
		vec3 trNormal = (_instanceDatum[i].normal).xyz();
		trNormal = (normalM * _instanceDatum[i].normal).xyz();
		trNormal.Normalize();
		vec3 eyePos = (mv *_instanceDatum[i].position).xyz();
		eyePos.Normalize();
		eyePos = eyePos * -1;
	    //����ü�
		if((eyePos).Dot(trNormal)>0)
		{
			//�����µ�vao
			memcpy(mirrorVAO+k*21,_VAOBuffer+i*21,sizeof(float)*21);
			//��Ƭid��
			mirrorVAO[k*21 +6] = k;
			mirrorVAO[k*21 +13] = k;
			mirrorVAO[k*21 +20] = k;
			k++;
			instances.push_back(_instanceDatum[i]);			
		}
	}
	if (instances.size() > 0)
	{
		for(int j=0; j<instances.size()*3; j++)
				indices.push_back(j);
		_mirrorVAO->SetVertexArray(indices.size()*7*sizeof(float),mirrorVAO);
		_mirrorVAO->SetElementArray(GL_UNSIGNED_INT,indices.size()*sizeof(uint),&indices[0]);
		_mirrorDataBuffer->SetBufferData(instances.size()*sizeof(InstanceData),&instances[0].position[0]);
		//_mirrorBuffer->BindBuffer(GL_RGBA32F,_mirrorDataBuffer);		
		//_mirrorUB->SetBufferData(instances.size() *sizeof(InstanceData),&instances[0].position[0]);

		delete[] mirrorVAO;
		return indices.size()/3;
	}
	else
	{
		return 0;
	}
}
void GIMApp::InitShadersOld()
{
	_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl","../../CommonSampleFiles/shaders/object.frag.glsl");
	_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 

	_mirrorShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/stencilBuildingO.vert.glsl","../../CommonSampleFiles/shaders/stencilBuilding.frag.glsl");

	_giShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/giVA.vert.glsl","../../CommonSampleFiles/shaders/gi.frag.glsl");

	_mirrorTexShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/mirrorTexture.vert.glsl","../../CommonSampleFiles/shaders/mirrorTexture.frag.glsl");
	_testShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/testm.vert.glsl","../../CommonSampleFiles/shaders/testm.frag.glsl");
}
void GIMApp::InitShaders()
{
	InitShadersOld();
	return;
	_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl","../../CommonSampleFiles/shaders/object.frag.glsl");
	_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 

	_mirrorShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/stencilBuilding.vert.glsl","../../CommonSampleFiles/shaders/stencilBuilding.geom.glsl","../../CommonSampleFiles/shaders/stencilBuilding.frag.glsl");

	_giShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/gi.vert.glsl","../../CommonSampleFiles/shaders/gi.frag.glsl");

	_mirrorTexShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/mirrorTexture.vert.glsl","../../CommonSampleFiles/shaders/mirrorTexture.frag.glsl");
}

void GIMApp::Display()
{	
	DisplayOld();	
	return;
	
	// Start timing this frame draw
	_frameRate->StartFrame();
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

#ifdef DEBUG
	_gpuTimer->Start();
	_cpuTimer->Start();
#endif

	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, _counterBuffer);
	GLuint *counters = (GLuint*)glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,	GL_WRITE_ONLY);
	counters[0] = 0;
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	IGLUOBJReader::Ptr mirrorMesh = _objReaders[_mirrorId];

	int instances = 0; /*= mirrorMesh->GetTriangleCount();*/
	/*int instances = UpdateMirrorVAO();*/

	_mirrorStencilFBO->Bind();
	_mirrorStencilFBO->Clear();
	_mirrorShader->Enable();

	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, _counterBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,_SSBO);	
	_mirrorShader["project"] = _camera->GetProjectionMatrix();
	_mirrorShader["model"] = _objTransforms[_mirrorId];
	_mirrorShader["view"] = _camera->GetViewMatrix();
	_mirrorVAO->DrawElements(GL_TRIANGLES,mirrorMesh->GetTriangleCount()*3);
	_mirrorShader->Disable();
	_mirrorStencilFBO->Unbind();
	//Now map the buffer and read it
	GLuint * valid = (GLuint*)glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,	GL_READ_ONLY);	
	instances = *valid;
	//printf("%d\n",instances);
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER,_SSBO);	
	//
	//GLuint * ssbo= (GLuint*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 
	//	sizeof(int)*mirrorMesh->GetTriangleCount(), 
 //                   GL_MAP_WRITE_BIT| GL_MAP_READ_BIT);
	//
	//for(int i=0; i<instances; i++)
	//{
	//	if (ssbo[i] == 2048)
	//	printf("when i=%d ssbo[i]=%d\n",i,ssbo[i]);
	//}
	//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	//IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], 0 );
	//return;
#ifdef DEBUG
	printf("stencil building gpu: %lf cpu: %lf \n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif
	
	if (instances > 0)
	{
		_mirrorFBO->Bind();
		_mirrorFBO->Clear();
		_giShader->Enable();
		_giShader["project"]         = _camera->GetProjectionMatrix();	
		_giShader["view"]            = _camera->GetViewMatrix();
		_giShader["lightIntensity" ] = float(1);
		_giShader["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_giShader["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		/*_giShader["lightPos"] = _lightPos;*/
		_giShader["lightColor"] = _lightColor;	
		_giShader["resX"] = _camera->GetResX();
		_giShader["resY"] = _camera->GetResY();
		_giShader["stencilTexture"] = _mirrorStencilFBO[0];
		 _giShader["mirrorModel"] = _objTransforms[_mirrorId];
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,_SSBO);
		for(int i=0; i<_objReaders.size(); i++)
		{
			if (i != _mirrorId)
			{
				_giShader["model"]           = _objTransforms[i];				
				glDisable(GL_CULL_FACE);//������ƽ��ñ����޳�
				glEnable(GL_DEPTH_TEST);
				_objReaders[i]->DrawMultipleInstances(_giShader,_mirrorBuffer,instances);
				glEnable(GL_CULL_FACE);				
			}
		}
		_giShader->Disable();
		_mirrorFBO->Unbind();
	//	GLuint * ssbo= (GLuint*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 
	//	sizeof(int)*mirrorMesh->GetTriangleCount(), 
 //                   GL_MAP_WRITE_BIT| GL_MAP_READ_BIT);
	//
	//for(int i=0; i<2; i++)
	//{
	//	
	//	printf("when i=%d ssbo[i]=%d %d\n",i,ssbo[i],ssbo[ssbo[i]]);

	//}
	//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  /*  	IGLUDraw::Fullscreen( _mirrorFBO[IGLU_COLOR0], 0 );
		return;*/
#ifdef DEBUG
		printf("GI Drawing gpu: %lf cpu: %lf \n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif
		
		
		_mirrorTexShader->Enable();
		_mirrorTexShader["project"]         = _camera->GetProjectionMatrix();
		_mirrorTexShader["modelview"]            = _camera->GetViewMatrix() * _objTransforms[_mirrorId];
		//_mirrorTexShader["lightIntensity" ] = float(1);
		//_mirrorTexShader["lightPos"] = _lightPos;
		//_mirrorTexShader["lightColor"] = _lightColor;	
		_mirrorTexShader["mtexture"] = _mirrorFBO[0];

		_mirrorVAO->DrawElements(GL_TRIANGLES,mirrorMesh->GetTriangleCount()*3);
		_mirrorTexShader->Disable();
#ifdef DEBUG
		printf("mirror texturing: %lf cpu: %lf \n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif
	}
	_objShader->Enable();
	_objShader["project"]         = _camera->GetProjectionMatrix();
	_objShader["view"]            = _camera->GetViewMatrix();
	_objShader["lightIntensity" ] = float(1);
	_objShader["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
	_objShader["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
	//_objShader["lightPos"] = _lightPos;
	_objShader["lightColor"] = _lightColor;	
	for(int i=0; i<_objReaders.size(); i++)
	{
		if (i != _mirrorId)
		{
			_objShader["model"]           = _objTransforms[i];			
			_objReaders[i]->Draw(_objShader);			
		}
	}
	_objShader->Disable();
	//IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], IGLU_DRAW_FLIP_Y );
	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}


void GIMApp::DisplayOld()
{

	_frameRate->StartFrame();
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

#ifdef DEBUG
	_gpuTimer->Start();
	_cpuTimer->Start();
#endif
	IGLUOBJReader::Ptr mirrorMesh = _objReaders[_mirrorId];
	// Start timing this frame draw
	
	/*int instances = mirrorMesh->GetTriangleCount();*/
	int instances = UpdateMirrorVAO();
#ifdef DEBUG
	printf("UpdateMirrorVAO:%lf  cpu:%lf\n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif
	
	_mirrorStencilFBO->Bind();
	_mirrorStencilFBO->Clear();
	_mirrorShader->Enable();

	_mirrorShader["project"] = _camera->GetProjectionMatrix();
	_mirrorShader["model"] = _objTransforms[_mirrorId];
	_mirrorShader["view"] = _camera->GetViewMatrix();
	
	_mirrorVAO->DrawElements(GL_TRIANGLES,instances*3);
	_mirrorShader->Disable();
	_mirrorStencilFBO->Unbind();
#ifdef DEBUG
	printf("build stencil gpu:%lf  cpu:%lf\n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif
	////
	/*IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], 0 );
	return;*/
	//finish = clock();
	//printf("stencil building %f\n", (double)(finish- start)/CLOCKS_PER_SEC);
	/*_testShader->Enable();
	_testShader["project"]         = _camera->GetProjectionMatrix();	
	_testShader["view"]            = _camera->GetViewMatrix();
	_testShader["model"]            = IGLUMatrix4x4::Identity();
	_testShader["stencilTexture"] = _mirrorStencilFBO[0];
	_mirrorVAO->DrawElements(GL_TRIANGLES,instances*3);
	_testShader->Disable();
	return;*/

	if (instances > 0)
	{
		_mirrorFBO->Bind();
		_mirrorFBO->Clear();
		_giShader->Enable();
		_giShader["project"]         = _camera->GetProjectionMatrix();	
		_giShader["view"]            = _camera->GetViewMatrix();
		_giShader["lightIntensity" ] = float(1);
		_giShader["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_giShader["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		/*_giShader["lightPos"] = _lightPos;*/
		_giShader["mirrorModel"] = _objTransforms[_mirrorId];
		_giShader["lightColor"] = _lightColor;	
		_giShader["resX"] = _camera->GetResX();
		_giShader["resY"] = _camera->GetResY();
		_giShader["stencilTexture"] = _mirrorStencilFBO[0];
		/*int uniformBlkIdx = glGetUniformBlockIndex( _giShader->GetProgramID(), "InstanceData" );
		 glBindBufferBase( GL_UNIFORM_BUFFER, uniformBlkIdx, _mirrorUB->GetBufferID() );*/
		for(int i=0; i<_objReaders.size(); i++)
		{
			if (i != _mirrorId)
			{
				_giShader["model"]           = _objTransforms[i];				
				glDisable(GL_CULL_FACE);//������ƽ��ñ����޳�
				glEnable(GL_DEPTH_TEST);
				_objReaders[i]->DrawMultipleInstances(_giShader,_mirrorDataBuffer->GetBufferID(), instances);
				glEnable(GL_CULL_FACE);
				
			}
		}
		_giShader->Disable();
		_mirrorFBO->Unbind();
  //  	IGLUDraw::Fullscreen( _mirrorFBO[IGLU_COLOR0], 0 );
		//return;
#ifdef DEBUG
		printf("GI��Drawing GPU:%f  CPU: %f\n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif
	    

		_mirrorTexShader->Enable();
		_mirrorTexShader["project"]         = _camera->GetProjectionMatrix();
		_mirrorTexShader["modelview"]            = _camera->GetViewMatrix() * _objTransforms[_mirrorId];
		//_mirrorTexShader["lightIntensity" ] = float(1);
		//_mirrorTexShader["lightPos"] = _lightPos;
		//_mirrorTexShader["lightColor"] = _lightColor;	
		_mirrorTexShader["mtexture"] = _mirrorFBO[0];

		_mirrorVAO->DrawElements(GL_TRIANGLES, instances*3);
		_mirrorTexShader->Disable();

#ifdef DEBUG
		printf("Mirror��texturing gpu: %f cpu: %f \n", _gpuTimer->Tick(), _cpuTimer->Tick());
#endif

	}

	_objShader->Enable();
	_objShader["project"]         = _camera->GetProjectionMatrix();
	_objShader["view"]            = _camera->GetViewMatrix();
	_objShader["lightIntensity" ] = float(1);
	_objShader["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
	_objShader["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
	//_objShader["lightPos"] = _lightPos;
	_objShader["lightColor"] = _lightColor;	
	for(int i=0; i<_objReaders.size(); i++)
	{
		if (i != _mirrorId)
		{
			_objShader["model"]           = _objTransforms[i];			
			_objReaders[i]->Draw(_objShader);			
		}
	}
	_objShader->Disable();

	//printf("Plain��Drawing %f\n", (double)(finish- start)/CLOCKS_PER_SEC);
	//IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], IGLU_DRAW_FLIP_Y );
	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );

}