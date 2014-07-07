#include "VORApp.h"
#include "CudaFunctions.h"
namespace OGL
{
const size_t MaxELEMENT = 1024*1024*3;
	using namespace iglu;
	void VORApp::InitBuffer()
	{
		_indirectDrawBuffer = new IGLUBuffer(iglu::IGLU_DRAW_INDIRECT);
		_indirectDrawBuffer->SetBufferData(sizeof(DrawElementsIndirectCommand)*MaxELEMENT,NULL,IGLU_DYNAMIC);
		//DrawElementsIndirectCommand * cmd = (DrawElementsIndirectCommand *)_indirectDrawBuffer->Map(IGLU_WRITE);	
		//		
		//GLuint count = _objReaders[0]->GetTriangleCount()*3;
		//GLuint vcount = _objReaders[0]->GetVertecies().size();
		//for (unsigned i = 0; i < 2; i++)
		//{
		//	cmd[i].firstIndex=0;
		//	cmd[i].count = count;
		//	cmd[i].instanceCount = 1;
		//	cmd[i].baseInstance = i;
		//	cmd[i].baseVertex = 0;
		//}

		//_indirectDrawBuffer->Unmap();
		//
		//实时更新的attrib Buffer
		_attribBuffer= new IGLUBuffer();
		_attribBuffer->SetBufferData(sizeof(uint)*MaxELEMENT,NULL,IGLU_STREAM);
	/*	int* attrib = (int*)_attribBuffer->Map(IGLU_WRITE);
		attrib[0] = 0;
		attrib[1] = 1;
		_attribBuffer->Unmap();*/

		_mirrorBuffer = new IGLUBuffer();
		_mirrorBuffer->SetBufferData(sizeof(float)*_instanceData.size(),&_instanceData[0]);
		_instanceDataTex = new IGLUTextureBuffer();
		_instanceDataTex->BindBuffer(GL_RGBA32F,_mirrorBuffer);
	}
	void VORApp::InitOGLCuda()
	{
		cudaDeviceProp prop;
		int dev;

		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 1;
		prop.minor = 0;
		cudaChooseDevice(&dev, &prop);
		cudaGLSetGLDevice(dev);

		_attribBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&_resource, _attribBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_attribBuffer->Unbind();
		_indirectDrawBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&_elemSource, _indirectDrawBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_indirectDrawBuffer->Unbind();
	}
	size_t VORApp::VirtualFrustumsCulling()
	{
		/*vec3 eye = GetCamera()->GetEye();*/
		vec3 eye(-5,0,0);
		//创建虚视锥，并进行裁剪，获得所有虚视锥中的三角形索引，将结果保存在GPU纹理内存中
		size_t size = cuda::VirtualFrustumCulling(_triSize,_mirrorTransforms.size(),eye,150,
			&_mirrorPos[0],&_mirrorTransId[0],&_mirrorTransforms[0],_mirrorPos.size()/9);
		
		//更新OPENGL用于绘制的虚顶点索引
		UpdateVirtualObject(size);
		return size;
	}
	void VORApp::UpdateVirtualObject(size_t size)
	{
		//更新虚物体索引buffer大小，并注册cuda互访问
		uint* cmdPtr;
		float * attribPtr;

		//map 
		size_t  bufferSize;
		cudaGraphicsMapResources(1, &_resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&attribPtr, &bufferSize, _resource);
		cudaGraphicsMapResources(1, &_elemSource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&cmdPtr, &bufferSize, _elemSource);

		cuda::UpdateVirtualObject(attribPtr,cmdPtr,size);

		//unmap
		cudaGraphicsUnmapResources(1, &_resource, NULL);
		cudaGraphicsUnmapResources(1, &_elemSource, NULL);
	}
	void VORApp::InitScene()
	{
		for (int i=0; i<_sceneData->getObjects().size(); i++)
		{
			SceneObject* obj = _sceneData->getObjects()[i];
			switch (obj->getType())
			{
			case SceneObjType::mesh:
				{
					ObjModelObject* mesh = (ObjModelObject*)obj;

					IGLUOBJReader::Ptr objReader;

					if (mesh->getUnitizeFlag())
					{
						objReader = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(), IGLU_OBJ_UNITIZE | IGLU_OBJ_COMPACT_STORAGE);
					}
					else
					{
						objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(),IGLU_OBJ_COMPACT_STORAGE);
					}

					glm::mat4 trans = (mesh->getTransform());					
					IGLUMatrix4x4 model(&trans[0][0]);

					if (!obj->getMaterialName().compare("mirror"))
					{
						_mirrorObjs.push_back(objReader);
						_mirrorTransforms.push_back(model);
					}
					else
					{
						_objReaders.push_back(objReader);
						_objTransforms.push_back(model);
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

		InitMirrorData();

		//对于场景中所有非镜面物体，创建bvh，返回三角形数
		_triSize = cuda::BuildBvh(&_objReaders[0], &_objTransforms[0], _objReaders.size());

		InitBuffer();
		InitOGLCuda();
		InitAttribute();
	}

	void VORApp::InitAttribute()
	{
		
		glBindVertexArray(_objReaders[0]->GetVertexArray()->GetArrayID());

		

		_objReaders[0]->GetVertexArray()->GetVertexBuffer()->Bind();

		bool vertAvail = _objReaders[0]->HasVertices();   // && (shader[ iglu::IGLU_ATTRIB_VERTEX ] != 0);
		bool normAvail = _objReaders[0]->HasNormals();    // && (shader[ iglu::IGLU_ATTRIB_NORMAL ] != 0);
		bool texAvail  = _objReaders[0]->HasTexCoords();  // && (shader[ iglu::IGLU_ATTRIB_TEXCOORD ] != 0);
		bool matlAvail = _objReaders[0]->HasMatlID();     // && (shader[ iglu::IGLU_ATTRIB_MATL_ID ] != 0);
		bool objectAvail = _objReaders[0]->HasObjectID();
		unsigned stride = _objReaders[0]->GetArrayBufferStride();
		unsigned vertOff    = 2 * sizeof( float );
		unsigned normOff    = (normAvail ? 5 : 0) * sizeof( float );
		unsigned texOff     = (texAvail ? (normAvail ? 8 : 5) : 0) * sizeof( float );
		//   We'll have 1 float for a material ID
		//   We'll have 1 float for an object ID
		//   We'll have 3 floats (x,y,z) for each of the 3 verts of each triangle 
		//   We'll have 3 floats (x,y,z) for each of the 3 normals of each triangle
		//   We'll have 2 floats (u,v) for each of the 3 texture coordinates of each triangle
		if (vertAvail)
		{
			glEnableVertexAttribArray( 0 );
			glVertexAttribPointer( 0, 3, GL_FLOAT, false, stride, (char *)NULL + (vertOff) );
		}
		if (normAvail)
		{
			glEnableVertexAttribArray( 1 );
			glVertexAttribPointer( 1, 3, GL_FLOAT, false, stride, (char *)NULL + (normOff) );
		}
		if (texAvail)
		{
			glEnableVertexAttribArray( 2 );
			glVertexAttribPointer( 2, 2, GL_FLOAT, false, stride, (char *)NULL + (texOff) );
		}
		if (matlAvail)
		{
			glEnableVertexAttribArray( 3 );
			glVertexAttribPointer( 3, 1, GL_FLOAT, false, stride, (char *)NULL + (0) );
		}
		if (objectAvail)
		{
			glEnableVertexAttribArray( 4 );
			glVertexAttribPointer( 4, 1, GL_FLOAT, false, stride, (char *)NULL + (4) );
		}
	

		_attribBuffer->Bind();		
		glVertexAttribPointer( 10, 1, GL_FLOAT, false, 0, (char *)NULL + (0) );
		glVertexAttribDivisor(10, 1);
		glEnableVertexAttribArray( 10);

	
		
	
		
	}
	void VORApp::InitShaders()
	{
		_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl","../../CommonSampleFiles/shaders/object.frag.glsl");
		_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 
		_simpleShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/simple.vert.glsl","../../CommonSampleFiles/shaders/simple.frag.glsl");

		_shaders.push_back(_objShader);
		_shaders.push_back(_simpleShader);

		_shaders.push_back(new IGLUShaderProgram("../../CommonSampleFiles/shaders/vor.vert.glsl","../../CommonSampleFiles/shaders/vor.frag.glsl"));
	}
	void VORApp::InitMirrorData()
	{
		unsigned size = 0;
		unsigned *offsets = new unsigned[_mirrorObjs.size()+1];
	
		for(int i=0; i<_mirrorObjs.size(); i++)
		{
			offsets[i] = size;
			size += _mirrorObjs[i]->GetTriangleCount();

		}
		_mirrorPos.resize(size*9);
		_mirrorTransId.resize(size);
		_instanceData.resize(size*8);
		for( int i=0; i<_mirrorObjs.size(); i++)
		{
			float* matrix = _mirrorTransforms[i].GetDataPtr();
			std::vector<iglu::vec3> vertices = _mirrorObjs[i]->GetVertecies();
			std::vector<iglu::IGLUOBJTri*> triangles = _mirrorObjs[i]->GetTriangles();
			unsigned offset = offsets[i]*9;
			unsigned offset2 = offsets[i]*8;
			for(int j=0; j<triangles.size(); j++)
			{			
				vec3 p[3];
				
				for( int k=0; k<3; k++)
				{
					p[k] = vertices[triangles[j]->vIdx[k]];
					memcpy(&_mirrorPos[offset+3*k],vertices[triangles[j]->vIdx[k]].GetConstDataPtr(),12);
				}
				memcpy(&_instanceData[offset2],p[0].GetConstDataPtr(),12);
				_instanceData[offset2+3] = 1.0f;
				vec3 normal = (p[2]-p[1]).Cross(p[0]-p[1]);
				normal.Normalize();
				memcpy(&_instanceData[offset2+4],normal.GetConstDataPtr(),12);
				_instanceData[offset2+7] = 0.0f;
				_mirrorTransId[offsets[i]] = i;
				offset += 9;
				offset2 += 8;
			}
		}
		delete[] offsets;
	}
	void VORApp::Display()
	{
		
		// Start timing this frame draw
		_frameRate->StartFrame();
		size_t size = VirtualFrustumsCulling();
	
		//size_t size = _objReaders[0]->GetTriangleCount();
		// Clear the screen
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		for(int i=0; i<_objReaders.size(); i++)
		{

			/*_shaders[0]->Enable();
			_shaders[0]["project"]         = _camera->GetProjectionMatrix();
			_shaders[0]["model"]           = _objTransforms[i];
			_shaders[0]["view"]            = _camera->GetViewMatrix();
			_shaders[0]["lightIntensity" ] = float(1);
			_shaders[0]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
			_shaders[0]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;			
			_shaders[0]["lightColor"] = _lightColor;	*/
			//_objReaders[i]->Draw(_shaders[0]);

			//_shaders[1]->Enable();
			//_shaders[1]["vcolor"] = vec4(0,0,1,0);
			//_shaders[1]["mvp"] = _camera->GetProjectionMatrix()*_camera->GetViewMatrix()*_objTransforms[i];
			_shaders[2]->Enable();
			_shaders[2]["project"] = _camera->GetProjectionMatrix();	
			_shaders[2]["view"] = _camera->GetViewMatrix();
			_shaders[2]["model"]  = _objTransforms[i];
			_shaders[2]["mirrorModel"] = _objTransforms[i];
			_shaders[2]["InstanceData"] = _instanceDataTex;
			/*_shaders[2]["lightIntensity" ] = float(1);
			_shaders[2]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
			_shaders[2]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;*/
			IGLUVertexArray::Ptr vao = _objReaders[i]->GetVertexArray();
			vao->Bind();
			
			vao->GetElementArray()->Bind();
			
			
			_indirectDrawBuffer->Bind();
			glMultiDrawElementsIndirect(	GL_TRIANGLES, 	GL_UNSIGNED_INT, 	NULL, 	size, 	0);
			_indirectDrawBuffer->Unbind();
			
		
			vao->GetElementArray()->Unbind();
			
			//
			vao->Unbind();
			/*_shaders[0]->Disable();*/
			_shaders[2]->Disable();
			//_shaders[1]->Disable();
		}
		// Draw the framerate on the screen
		char buf[32];
		sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
		IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
	}
}