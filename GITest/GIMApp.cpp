#include "GIMApp.h"
#include<glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
struct InstanceData
{
	InstanceData():position(vec4(0,0,0,1)),normal(vec4(0.f)){}
	vec4 position;
	vec4 normal;
};



GIMApp::~GIMApp()
{	
	delete _mirrorBuffer;
	delete _mirrorVAO;
}
void GIMApp::InitBuffer()
{
	IGLUOBJReader::Ptr _mirrorMesh = _objReaders[_mirrorId];
	//保存镜面的法线和点
	int TriNum = _mirrorMesh->GetTriangleCount();	
	vector<InstanceData> instances(TriNum);
	uint *indexBuffer =(uint*) malloc(TriNum*3*sizeof(uint));

	//绘制镜面蒙版和镜面时的vao
	_mirrorVAO = new IGLUVertexArray();	

	//每个三角形的顶点有vertex 3,normal 3,index共7个浮点数
	float* VAOBuffer = (float*)malloc(TriNum*3*7*sizeof(float));

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
			instances[i].position[j] = vertices[tris[i]->vIdx[0]].GetElement(j);	
			v[j] = vertices[tris[i]->vIdx[j]];
			for(int k=0; k<3; k++)
			{
				VAOBuffer[i*21+j*7+k] = vertices[tris[i]->vIdx[j]].GetElement(k);
			}
			for(int k=0; k<3; k++)
			{				
				VAOBuffer[i*21+j*7+k+3] = normals[tris[i]->nIdx[j]].GetElement(k);
			}
			VAOBuffer[i*21+j*7+6] = i;
		}	
		vec3 faceNormal = (v[2]-v[1]).Cross(v[0]-v[1]);
		faceNormal.Normalize();
		_mirrorNormals.push_back(faceNormal);
		for(int j=0; j<3; j++)
		{		
			instances[i].normal[j] = faceNormal.GetElement(j);
		}		
	}

	IGLUMatrix4x4 model = _objTransforms[_mirrorId];
	IGLUMatrix4x4 normalM = model.Transpose();
	glm::mat4 mat = glm::make_mat4(normalM.GetDataPtr());
	mat = glm::inverse(mat);
	normalM = IGLUMatrix4x4(&mat[0][0]);
	for( int i=0; i< TriNum; i++)
	{
		instances[i].position = model * instances[i].position;
		instances[i].normal = vec4((normalM * instances[i].normal).xyz(),0);

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
	float* tmp = &instances[0].position[0];
	/*for( int i=0; i< TriNum*2*4; i+=8)
	{
	printf("tmpbuffer\n");
	printf("v %f, %f, %f, %f\n n %f, %f, %f, %f\n",tmp[i],tmp[i+1],tmp[i+2],tmp[i+3],
	tmp[i+4],tmp[i+5],tmp[i+6],tmp[i+7]);
	}*/
	IGLUBuffer::Ptr dataBuffer = new IGLUBuffer();
	dataBuffer->SetBufferData(TriNum*2*4*sizeof(float),tmp);
	_mirrorBuffer = new IGLUTextureBuffer();
	_mirrorBuffer->BindBuffer(GL_RGBA32F,dataBuffer);


	_mirrorVAO->SetVertexArray(TriNum*3*7*sizeof(float),VAOBuffer);
	_mirrorVAO->SetElementArray(GL_UNSIGNED_INT,TriNum*3*sizeof(uint),indexBuffer);

	_mirrorVAO->EnableAttribute(0,3,GL_FLOAT,7*sizeof(float));
	_mirrorVAO->EnableAttribute(1,3,GL_FLOAT,7*sizeof(float),BUFFER_OFFSET(3*sizeof(float)));	
	_mirrorVAO->EnableAttribute(2,1,GL_FLOAT,7*sizeof(float),BUFFER_OFFSET(6*sizeof(float)));


	free (VAOBuffer);
	free(indexBuffer);

	_mirrorStencilFBO =  IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,true);
	_mirrorFBO = IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,true);

	int * ssboData = (int*)malloc(sizeof(int)*TriNum);
	for(int i=0; i< TriNum; i++)
	{
		ssboData[i] = i;
	}
	glGenBuffers(1,&_SSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,_SSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,sizeof(int)*TriNum,ssboData,GL_DYNAMIC_COPY);
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
				IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str());
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
	vector<uint> indices;
	vec3 viewDir = _camera->GetViewDirection();
	for(int i=0; i<_mirrorNormals.size(); i++)
	{
		vec4 trNormal = _camera->GetViewMatrix()*vec4(_mirrorNormals[i],0);

		if (i==1/*(trNormal.xyz()).Dot(viewDir) < 0*/)
		{
			for(int j=0; j<3; j++)
				indices.push_back(i*3+j);
		}
	}
	if (indices.size() > 0)
	{
		_mirrorVAO->SetElementArray(GL_UNSIGNED_INT,indices.size()*sizeof(uint),&indices[0]);
		return indices.size()/3;
	}
	else
	{
		return 0;
	}
}

void GIMApp::InitShaders()
{
	_objShader = new IGLUShaderProgram("shaders/object.vert.glsl","shaders/object.frag.glsl");
	_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 

	_mirrorShader = new IGLUShaderProgram("shaders/stencilBuilding.vert.glsl","shaders/stencilBuilding.geom.glsl","shaders/stencilBuilding.frag.glsl");

	_giShader = new IGLUShaderProgram("shaders/gi.vert.glsl","shaders/gi.frag.glsl");

	_mirrorTexShader = new IGLUShaderProgram("shaders/mirrorTexture.vert.glsl","shaders/mirrorTexture.frag.glsl");
}

void GIMApp::Display()
{
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, _counterBuffer);
	GLuint *counters = (GLuint*)glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,	GL_WRITE_ONLY);
	counters[0] = 0;
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	IGLUOBJReader::Ptr mirrorMesh = _objReaders[_mirrorId];
	// Start timing this frame draw
	_frameRate->StartFrame();
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

	//_mirrorShader["viewDir"] = _camera->GetViewDirection();
	_mirrorVAO->DrawElements(GL_TRIANGLES,mirrorMesh->GetTriangleCount()*3);
	_mirrorShader->Disable();
	_mirrorStencilFBO->Unbind();
	//Now map the buffer and read it
	GLuint * valid = (GLuint*)glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,	GL_READ_ONLY);	
	instances = *valid;
	//printf("%d\n",instances);
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	
	/*glBindBuffer(GL_SHADER_STORAGE_BUFFER,_SSBO);	
	
	GLuint * ssbo= (GLuint*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 
		sizeof(int)*mirrorMesh->GetTriangleCount(), 
                    GL_MAP_WRITE_BIT| GL_MAP_READ_BIT);
	
	for(int i=0; i<instances; i++)
	{
		if (ssbo[i] != i)
		printf("when i=%d ssbo[i]=%d\n",i,ssbo[i]);
	}
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);*/
	/*IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], 0 );
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
		_giShader["lightColor"] = _lightColor;	
		_giShader["resX"] = _camera->GetResX();
		_giShader["resY"] = _camera->GetResY();
		_giShader["stencilTexture"] = _mirrorStencilFBO[0];
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,_SSBO);
		for(int i=0; i<_objReaders.size(); i++)
		{
			if (i != _mirrorId)
			{
				_giShader["model"]           = _objTransforms[i];				
				glDisable(GL_CULL_FACE);//镜面绘制禁用背面剔除
				glEnable(GL_DEPTH_TEST);
				_objReaders[i]->DrawMultipleInstances(_giShader,_mirrorBuffer,instances);
				glEnable(GL_CULL_FACE);
				
			}
		}
		_giShader->Disable();
		_mirrorFBO->Unbind();
    /*	IGLUDraw::Fullscreen( _mirrorFBO[IGLU_COLOR0], 0 );
		return;*/


		_mirrorTexShader->Enable();
		_mirrorTexShader["project"]         = _camera->GetProjectionMatrix();
		_mirrorTexShader["modelview"]            = _camera->GetViewMatrix() * _objTransforms[_mirrorId];
		//_mirrorTexShader["lightIntensity" ] = float(1);
		//_mirrorTexShader["lightPos"] = _lightPos;
		//_mirrorTexShader["lightColor"] = _lightColor;	
		_mirrorTexShader["mtexture"] = _mirrorFBO[0];

		_mirrorVAO->DrawElements(GL_TRIANGLES,mirrorMesh->GetTriangleCount()*3);
		_mirrorTexShader->Disable();

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