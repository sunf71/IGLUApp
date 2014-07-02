#include "VORApp.h"
#include "CudaFunctions.h"
namespace OGL
{
	using namespace iglu;
	void VORApp::InitBuffer()
	{
		/*_indirectDrawBuffer = new IGLUBuffer(iglu::IGLU_DRAW_INDIRECT);
		_indirectDrawBuffer->SetBufferData(sizeof(DrawElementsIndirectCommand)*_triSize,NULL,IGLU_DYNAMIC);*/

		_elemBuffer = new IGLUBuffer(IGLU_ELEMENT_ARRAY);
		_elemBuffer->SetBufferData(sizeof(uint)*_triSize*3,_objReaders[0]->GetElementArrayData(),IGLU_DYNAMIC);
		
		//实时更新的element buffer
		_vElemBuffer= new IGLUBuffer(IGLU_ELEMENT_ARRAY);
		_vElemBuffer->SetBufferData(sizeof(uint)*_triSize*3,NULL,IGLU_STREAM);
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

		_elemBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&_resource, _elemBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_elemBuffer->Unbind();
		_vElemBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&_elemSource, _vElemBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_vElemBuffer->Unbind();
	}
	size_t VORApp::VirtualFrustumsCulling()
	{
		/*vec3 eye = GetCamera()->GetEye();*/
		vec3 eye(-5,0,0);
		//创建虚视锥，并进行裁剪，获得所有虚视锥中的三角形索引，将结果保存在GPU纹理内存中
		size_t size = cuda::VirtualFrustumCulling(_triSize,eye,150,&_mirrorObjs[0],&_mirrorTransforms[0],_mirrorObjs.size(),NULL,NULL);
		
		//更新OPENGL用于绘制的虚顶点索引
		UpdateVirtualObject(size);
		return size;
	}
	void VORApp::UpdateVirtualObject(size_t size)
	{
		//更新虚物体索引buffer大小，并注册cuda互访问
		uint* inPtr,*outPtr;
		_vElemBuffer->Bind();
		_vElemBuffer->SetBufferData(sizeof(uint)*size*3,NULL,IGLU_STREAM);
		cudaGraphicsGLRegisterBuffer(&_elemSource, _vElemBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_vElemBuffer->Unbind();

		//map 
		size_t  bufferSize;
		cudaGraphicsMapResources(1, &_resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&inPtr, &bufferSize, _resource);
		cudaGraphicsMapResources(1, &_elemSource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&outPtr, &bufferSize, _elemSource);

		cuda::UpdateVirtualObject(inPtr,outPtr,size);

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

		//对于场景中所有非镜面物体，创建bvh，返回三角形数
		_triSize = cuda::BuildBvh(&_objReaders[0], &_objTransforms[0], _objReaders.size());

		InitBuffer();
		InitOGLCuda();
		InitAttribute();
	}

	void VORApp::InitAttribute()
	{
		_objReaders[0]->GetVertexArray()->Bind();
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
	}
	void VORApp::InitShaders()
	{
		_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl","../../CommonSampleFiles/shaders/object.frag.glsl");
		_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 
		_simpleShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/simple.vert.glsl","../../CommonSampleFiles/shaders/simple.frag.glsl");

		_shaders.push_back(_objShader);
		_shaders.push_back(_simpleShader);
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

			_shaders[1]->Enable();
			_shaders[1]["vcolor"] = vec4(0,0,1,0);
			_shaders[1]["mvp"] = _camera->GetProjectionMatrix()*_camera->GetViewMatrix()*_objTransforms[i];
			
			IGLUVertexArray::Ptr vao = _objReaders[i]->GetVertexArray();
			vao->Bind();
			_vElemBuffer->Bind();
			//_elemBuffer->Bind();
			
			glDrawElements(GL_TRIANGLES,size*3,GL_UNSIGNED_INT,NULL);
			_vElemBuffer->Unbind();
			//_elemBuffer->Unbind();
			//
			vao->Unbind();
			/*_shaders[0]->Disable();*/
			_shaders[1]->Disable();
		}
		// Draw the framerate on the screen
		char buf[32];
		sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
		IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
	}
}