#include "VORApp.h"
#include "CudaFunctions.h"
#include "bvh/frustum.h"
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

		
		_mirrorStencilFBO =  IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,false);
		_mirrorFBO = IGLUFramebuffer::Create(GL_RGBA16F,_winWidth,_winHeight,true,true);
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
		cudaGraphicsGLRegisterBuffer(&_attriRes, _attribBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_attribBuffer->Unbind();
		_indirectDrawBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&_cmdRes, _indirectDrawBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_indirectDrawBuffer->Unbind();

		//镜面elementbufer const		
		IGLUBuffer::Ptr mirrorElementBuffer = _mirrorObjs[0]->GetVertexArray()->GetElementArray();
		mirrorElementBuffer->Bind();	
		cudaGraphicsGLRegisterBuffer(&_mCEleRes, mirrorElementBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		mirrorElementBuffer->Unbind();

		//变化的镜面element buffer
		_mirrorElementBuffer->Bind();
		_mirrorElementBuffer->Bind();
		cudaGraphicsGLRegisterBuffer(&_mEleRes, _mirrorElementBuffer->GetBufferID(), cudaGraphicsMapFlagsNone);
		_mirrorElementBuffer->Unbind();

	}
	struct cResult
	{
		cResult(uint f,uint t):fid(f),tid(t)
		{}
		cResult(const cResult& c)
		{
			fid = c.fid;
			tid = c.tid;
		}
		uint fid;
		uint tid;
	};
	size_t VORApp::CPUVirtualFrustumsCulling(size_t& mirrorTrisize)
	{
		vec3 eye = GetCamera()->GetEyePos();
		std::vector<nih::TriFrustum> frustums;
		for(int i=0; i<_mirrorObjs[0]->GetTriangleCount()*9; i+=9)
		{
			//求5个平面方程
			nih::Vector3f p1(&_mirrorPos[i]);
			nih::Vector3f p2(&_mirrorPos[i+3]);
			nih::Vector3f p3(&_mirrorPos[i+6]);
			//视锥平面法线指向视锥外
			nih::plane_t pTri(p1,p2,p3);	
			nih::Vector3f veye(eye.GetConstDataPtr());
			float d  = pTri.distance(veye);
			//视点不能位于三角形平面法线那一侧
			if (d<= 0)
				continue;

			//求虚视点
			nih::Vector3f fNormal(pTri.a,pTri.b,pTri.c);
			float dir = nih::dot(veye-p1,fNormal);
			nih::Vector3f vEye = veye-fNormal*2.f*dir;
			nih::TriFrustum frustum;
			frustum.id = i/9;
			frustum.planes[0] = nih::plane_t(vEye,p2,p1);
			frustum.planes[1] = nih::plane_t(vEye,p3,p2);
			frustum.planes[2] = nih::plane_t(vEye,p1,p3);
			frustum.planes[3] =  nih::plane_t(p1,p3,p2);
			frustum.planes[4] = pTri;
			nih::Vector3f c = (p1+p2+p3)*1.f/3.f;
			float cosT = d/euclidean_distance(vEye,c);
			frustum.planes[4].d -= 150/cosT;		
			frustums.push_back(frustum);
		}
		mirrorTrisize = frustums.size();
		std::vector<cResult> cResults;
		for( int i=0; i<frustums.size(); i++)
		{
			for(int j=0; j<_objReaders[0]->GetTriangleCount(); j++)
			{
				uint i1 = 3*_objReaders[0]->GetElementArrayData()[3*j];
				uint i2 = 3*_objReaders[0]->GetElementArrayData()[3*j+1];
				uint i3 = 3*_objReaders[0]->GetElementArrayData()[3*j+2];
				nih::Vector3f p1(&(_objReaders[0]->GetVaoVerts()[i1]));
				nih::Vector3f p2(&_objReaders[0]->GetVaoVerts()[i2]);
				nih::Vector3f p3(&_objReaders[0]->GetVaoVerts()[i3]);
				if (nih::Intersect(frustums[i],p1,p2,p3)>0)
				{
					cResult r(i,j);
					cResults.push_back(r);
					//std::cout<<i<<":"<<j<<std::endl;
				}
			}
		}

		uint* element = new uint[frustums.size()];
		uint* mirrorElements = new uint[frustums.size()*3];
		uint* inPtr = _mirrorObjs[0]->GetElementArrayData();
		for(int i=0; i<frustums.size(); i++)
		{	
			element[i] = frustums[i].id;
			unsigned offset = 3*i;
			unsigned offset2 = frustums[i].id*3;
			mirrorElements[offset] = inPtr[offset2];
			mirrorElements[offset+1] = inPtr[offset2+1];
			mirrorElements[offset+2] = inPtr[offset2 +2];
		}

		DrawElementsIndirectCommand* cmd = new DrawElementsIndirectCommand[frustums.size()*10];
		for(int i=0; i<frustums.size(); i++)
		{
			for(int j=0; j<10; j++)
			{
				cmd[10*i+j].baseInstance = i;
				cmd[10*i+j].baseVertex = 0;
				cmd[10*i+j].count = 3;
				cmd[10*i+j].firstIndex = j;
				cmd[10*i+j].instanceCount = 1;
			}
		}
		_attribBuffer->SetBufferSubData(0,sizeof(uint)*frustums.size(),element);
		_indirectDrawBuffer->SetBufferSubData(0,sizeof(DrawElementsIndirectCommand)*10*frustums.size(),cmd);
		_mirrorElementBuffer->SetBufferSubData(0,sizeof(uint)*3*frustums.size(),mirrorElements);
		delete[] element;
		delete[] cmd;
		delete[] mirrorElements;
		return 10*frustums.size();
	}
	size_t VORApp::VirtualFrustumsCulling(size_t& mirrorTrisize)
	{
		//vec3 eye = GetCamera()->GetEye();
		vec3 eye = GetCamera()->GetEyePos();
		//vec3 eye(-5,0,0);
		

		unsigned* mCElePtr,*mElePtr;
		unsigned bufferSize;
		cudaGraphicsMapResources(1, &_mCEleRes, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&mCElePtr, &bufferSize, _mCEleRes);
		cudaGraphicsMapResources(1, &_mEleRes, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&mElePtr, &bufferSize, _mEleRes);
		mirrorTrisize = _mirrorPos.size()/9;
		//创建虚视锥，并进行裁剪，获得所有虚视锥中的三角形索引，将结果保存在GPU纹理内存中
		size_t size = cuda::VirtualFrustumCulling(_triSize,_mirrorTransforms.size(),eye,150,
			&_mirrorPos[0],&_mirrorTransId[0],&_mirrorTransforms[0],mirrorTrisize,mCElePtr,mElePtr);
		

		cudaGraphicsUnmapResources(1, &_mCEleRes, NULL);
		cudaGraphicsUnmapResources(1, &_mEleRes, NULL);

		//更新OPENGL用于绘制的虚顶点索引
		UpdateVirtualObject(size);
		return size;
	}
	void VORApp::UpdateVirtualObject(size_t size)
	{
		//更新虚物体索引buffer大小，并注册cuda互访问
		uint* cmdPtr, *mCElePtr, * mElePtr;
		float * attribPtr;

		//map 
		size_t  bufferSize;
		cudaGraphicsMapResources(1, &_attriRes, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&attribPtr, &bufferSize, _attriRes);
		cudaGraphicsMapResources(1, &_cmdRes, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&cmdPtr, &bufferSize, _cmdRes);
		

		cuda::UpdateVirtualObject(attribPtr,cmdPtr,size);

		//unmap
		cudaGraphicsUnmapResources(1, &_attriRes, NULL);
		cudaGraphicsUnmapResources(1, &_cmdRes, NULL);
		
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

					int params = mesh->getCompactFlag() ? IGLU_OBJ_COMPACT_STORAGE : 0;					
					params |=mesh->getUnitizeFlag() ?  IGLU_OBJ_UNITIZE : 0;
					objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(), params);

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
	void InitOBJReaderAttribute(IGLUOBJReader::Ptr reader)
	{
		reader->GetVertexArray()->Bind();
		reader->GetVertexArray()->GetVertexBuffer()->Bind();
		bool vertAvail = reader->HasVertices();   // && (shader[ iglu::IGLU_ATTRIB_VERTEX ] != 0);
		bool normAvail = reader->HasNormals();    // && (shader[ iglu::IGLU_ATTRIB_NORMAL ] != 0);
		bool texAvail  = reader->HasTexCoords();  // && (shader[ iglu::IGLU_ATTRIB_TEXCOORD ] != 0);
		bool matlAvail = reader->HasMatlID();     // && (shader[ iglu::IGLU_ATTRIB_MATL_ID ] != 0);
		bool objectAvail = reader->HasObjectID();
		unsigned stride = reader->GetArrayBufferStride();
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
	void VORApp::InitAttribute()
	{
		InitOBJReaderAttribute(_objReaders[0]);		
		_attribBuffer->Bind();		
		glVertexAttribPointer( 10, 1, GL_FLOAT, false, 0, (char *)NULL + (0) );
		glVertexAttribDivisor(10, 1);
		glEnableVertexAttribArray( 10);

		InitOBJReaderAttribute(_mirrorObjs[0]);
		size_t size = _mirrorObjs[0]->GetTriangleCount();
		float * index = new float[size*3];
		for(int i=0; i<size; i++)
			index[3*i]=index[3*i+1]=index[3*i+2]=i;
		IGLUBuffer::Ptr indexBuffer = new IGLUBuffer();
		indexBuffer->SetBufferData(size*sizeof(float)*3,index);
		indexBuffer->Bind();
		glVertexAttribPointer( 2, 1, GL_FLOAT, false, 0, (char *)NULL + (0) );
		glEnableVertexAttribArray(2);
		delete[] index;
	
		
	
		
	}
	void VORApp::InitShaders()
	{
		_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl","../../CommonSampleFiles/shaders/object.frag.glsl");
		_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 
		_simpleShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/simple.vert.glsl","../../CommonSampleFiles/shaders/simple.frag.glsl");

		_shaders.push_back(_objShader);
		_shaders.push_back(_simpleShader);

		_shaders.push_back(new IGLUShaderProgram("../../CommonSampleFiles/shaders/vor.vert.glsl","../../CommonSampleFiles/shaders/vor.frag.glsl"));

		_shaders.push_back(new IGLUShaderProgram("../../CommonSampleFiles/shaders/stencilBuildingO.vert.glsl","../../CommonSampleFiles/shaders/stencilBuildingO.frag.glsl"));

		_mirrorTexShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/mirrorTexture.vert.glsl","../../CommonSampleFiles/shaders/mirrorTexture.frag.glsl");
		_shaders.push_back(_mirrorTexShader);
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

		_mirrorElementBuffer = new IGLUBuffer(IGLU_ELEMENT_ARRAY);
		_mirrorElementBuffer->SetBufferData(sizeof(uint)*size*3,NULL,IGLU_STREAM);
		_mirrorPos.resize(size*9);
		_mirrorTransId.resize(size);
		_instanceData.resize(size*8);
		for( int i=0; i<_mirrorObjs.size(); i++)
		{
			float* matrix = _mirrorTransforms[i].GetDataPtr();
			vector<float> vertices = _mirrorObjs[i]->GetVaoVerts();
			uint* elements = _mirrorObjs[i]->GetElementArrayData();
			unsigned offset = offsets[i]*9;
			unsigned offset2 = offsets[i]*8;
			for(int j=0; j<vertices.size()/9; j++)
			{			
				vec3 p[3];
				
				for( int k=0; k<3; k++)
				{
					for(int m=0; m<3; m++)
						p[k][m] = vertices[elements[3*j+k]*3+m];
					memcpy(&_mirrorPos[offset+3*k],p[k].GetConstDataPtr(),12);
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
		// Clear the screen
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		
		
		size_t frustumSize = 0;
		size_t size = VirtualFrustumsCulling(frustumSize);
		//size_t size = CPUVirtualFrustumsCulling(frustumSize);
		//draw stencil
		{
			_mirrorStencilFBO->Bind();
			_mirrorStencilFBO->Clear();
			_shaders[3]->Enable();
			_shaders[3]["project"]         = _camera->GetProjectionMatrix();
			_shaders[3]["model"]           = _mirrorTransforms[0];
			_shaders[3]["view"]            = _camera->GetViewMatrix();
			_mirrorObjs[0]->GetVertexArray()->Bind();
			_mirrorElementBuffer->Bind();
			//_mirrorObjs[0]->GetVertexArray()->GetElementArray()->Bind();
			glDrawElements(GL_TRIANGLES,frustumSize*3,GL_UNSIGNED_INT,0);
			//_mirrorObjs[0]->GetVertexArray()->GetElementArray()->Unbind();
			_mirrorElementBuffer->Unbind();
			_mirrorObjs[0]->GetVertexArray()->Unbind();

			_shaders[3]->Disable();
			_mirrorStencilFBO->Unbind();
		/*	IGLUDraw::Fullscreen( _mirrorStencilFBO[IGLU_COLOR0], 0 );
			return;*/
		
		}
	
		_mirrorFBO->Bind();
		_mirrorFBO->Clear();
		glEnable(GL_DEPTH_TEST);
	
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
			_shaders[2]["lightIntensity" ] = float(1);
			_shaders[2]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
			_shaders[2]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
			_shaders[2]["lightColor"] = vec3(1.0);	
			_shaders[2]["resX"] = _camera->GetResX();
			_shaders[2]["resY"] = _camera->GetResY();
			_shaders[2]["stencilTexture"] = _mirrorStencilFBO[0];
			IGLUVertexArray::Ptr vao = _objReaders[i]->GetVertexArray();
			vao->Bind();			
			vao->GetElementArray()->Bind();			
			_indirectDrawBuffer->Bind();
			glMultiDrawElementsIndirect(	GL_TRIANGLES, 	GL_UNSIGNED_INT, 	NULL, 	size, 	0);
			_indirectDrawBuffer->Unbind();
			vao->GetElementArray()->Unbind();
			vao->Unbind();
			/*_shaders[0]->Disable();*/
			_shaders[2]->Disable();
			//_shaders[1]->Disable();
		}
		_mirrorFBO->Unbind();

		_mirrorTexShader->Enable();
		_mirrorTexShader["project"]         = _camera->GetProjectionMatrix();
		_mirrorTexShader["modelview"]            = _camera->GetViewMatrix() *_mirrorTransforms[0];
		//_mirrorTexShader["lightIntensity" ] = float(1);
		//_mirrorTexShader["lightPos"] = _lightPos;
		//_mirrorTexShader["lightColor"] = _lightColor;	
		_mirrorTexShader["mtexture"] = _mirrorFBO[0];

		_mirrorObjs[0]->GetVertexArray()->Bind();
		_mirrorElementBuffer->Bind();		
		glDrawElements(GL_TRIANGLES,frustumSize*3,GL_UNSIGNED_INT,0);		
		_mirrorElementBuffer->Unbind();
		_mirrorObjs[0]->GetVertexArray()->Unbind();
		_mirrorTexShader->Disable();

		for(int i=0; i<_objReaders.size(); i++)
		{
			_objShader->Enable();
			_objShader["project"]         = _camera->GetProjectionMatrix();
			_objShader["model"]           = _objTransforms[i];
			_objShader["view"]            = _camera->GetViewMatrix();
			_objShader["lightIntensity" ] = float(1);
			_objShader["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
			_objShader["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;			
			_objShader["lightColor"] = _lightColor;
			_objReaders[i]->GetVertexArray()->Bind();
			_objReaders[i]->GetVertexArray()->GetElementArray()->Bind();
			glDrawElements(GL_TRIANGLES,_objReaders[i]->GetTriangleCount()*3,GL_UNSIGNED_INT,0);
			_objReaders[i]->GetVertexArray()->GetElementArray()->Unbind();
			_objReaders[i]->GetVertexArray()->Unbind();
			_objShader->Disable();
		}
		// Draw the framerate on the screen
		char buf[32];
		sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
		IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
	}
}