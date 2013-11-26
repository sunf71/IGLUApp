#include "VFCIGLUApp.h"
#include "Frustum.h"
#include <hash_map>


template <typename T>
void CopyToVector(IGLUArray1D<T>& src, vector<T>& des)
{
	des.reserve(src.Size());
	for(int i=0; i<src.Size(); i++)
	{
		des.push_back(src[i]);
	}
}

void VFCIGLUApp::GetAABBs(IGLUOBJReader::Ptr &reader)
{
	int total = reader->GetTriangleCount();
	//获取每个三角形，计算AABB保存起来
	_triangleObjects.reserve(total);
	//_aabbs = vector<AaBox>(total);
	_bboxs = vector<BBox>(total);
	vector<IGLUOBJTri *> tris = reader->GetTriangles();
	vector<vec3> vertices = reader->GetVertecies();
	for(int i=0; i<total; i++)
	{
		IGLUOBJTri* tri = tris[i];
	    vec3 p1 = vertices[tri->vIdx[0]];
		vec3 p2 = vertices[tri->vIdx[1]];
		vec3 p3 = vertices[tri->vIdx[2]];

		//把三角形数据复制一份作为成员变量
		//不知为何回调函数中访问不了reader的数据，所以复制一份存起来
		IGLUOBJTri triangle(*tri);
		_triangles.push_back(triangle);
		Triangle *t = new Triangle(p1,p2,p3,i);
		_triangleObjects.push_back(t);

		//GetAABB(p1,p2,p3,_aabbs[i]);
		GetAABB(p1,p2,p3,_bboxs[i]);
	}

	//不知为何回调函数中访问不了reader的数据，所以复制一份存起来
	_texcoords = reader->GetTexCoords();
	_normals = reader->GetNormals();
	_vertices = reader->GetVertecies();
	_bvh = new BVH(&_triangleObjects);
	
}
string myhash(int vid,int nid,int tid)
{
	char result[200];
	sprintf(result,"%d,%d,%d",vid,nid,tid);
	return string(result);
}
void  VFCIGLUApp::UpdateReader( vector<int>& idxs, IGLUOBJReader::Ptr &reader)
{
	IGLUVertexArray::Ptr vao = reader->GetVertexArray();
	uint* elementArray = reader->GetElementArrayData();
	vector<uint> indices;
	for(int i=0; i<idxs.size(); i++)
	{
		for(int j=0; j<3; j++)
		{
			indices.push_back(elementArray[idxs[i]*3+j]);
		}
	}
	vao->SetElementArray(GL_UNSIGNED_INT,sizeof(uint)*indices.size(),&indices[0]);
}
void VFCIGLUApp::SetupNewVAO(IGLUVertexArray::Ptr& VAO, vector<int>& idxs, IGLUOBJReader::Ptr &reader)
{
	bool hasNormals = reader->HasNormals();
	bool hasTexCoords = reader->HasTexCoords();
	uint  numComponents = 1 + 1 + 3 + (hasNormals ? 3 : 0) + (hasTexCoords ? 2 : 0);
	int vertStride = numComponents * sizeof( float );
	int matlIdOff  = 0;
	int objectIdOff = 1; 
	int vertOff    = 2;
	int normOff    = (reader->HasNormals() ? 5 : 0);
	int texOff     = (reader->HasTexCoords() ? (reader->HasNormals() ? 8 : 5) : 0);

	
	//float* tmpBuf = (float*) malloc(numComponents*sizeof(float)*idxs.size()*3);
	vector<float> tmpBuf(numComponents*sizeof(float)*idxs.size()*3);
	hash_map<string,int> mapping;
	
	unsigned int *nElem = (unsigned int*) malloc(idxs.size()*3*sizeof(unsigned int));
	int k = 0;
	for(int i =0; i<idxs.size(); i++)
	{
		for(int j=0;j<3;j++)
		{
			int vid = _triangles[idxs[i]].vIdx[j];
			int nid = hasNormals ? _triangles[idxs[i]].nIdx[j] : 0;
			int tid = hasTexCoords ? _triangles[idxs[i]].tIdx[j] : 0;
			string id = myhash(vid,nid,tid);
			if (mapping.find(id) == mapping.end())
			{
				nElem[3*i+j] = k;
				mapping[id] = k;
							
				// Using our EXTREMELY naive approach...
				//   We'll have 1 float for a material ID
				//   We'll have 1 float for an object ID
				//   We'll have 3 floats (x,y,z) for each of the 3 verts of each triangle 
				//   We'll have 3 floats (x,y,z) for each of the 3 normals of each triangle
				//   We'll have 2 floats (u,v) for each of the 3 texture coordinates of each triangle
				//mat id
				tmpBuf[k*numComponents] = _triangles[idxs[i]].matlID;
				tmpBuf[k*numComponents+objectIdOff] = _triangles[idxs[i]].objectID;

				tmpBuf[k*numComponents+vertOff] = _vertices[_triangles[idxs[i]].vIdx[j]].X();
				tmpBuf[k*numComponents+vertOff+1] = _vertices[_triangles[idxs[i]].vIdx[j]].Y();				
				tmpBuf[k*numComponents+vertOff+2] = _vertices[_triangles[idxs[i]].vIdx[j]].Z();

				if (hasNormals)
				{
					tmpBuf[k*numComponents+normOff] = _normals[_triangles[idxs[i]].nIdx[j]].X();
					tmpBuf[k*numComponents+normOff+1] = _normals[_triangles[idxs[i]].nIdx[j]].Y();
					tmpBuf[k*numComponents+normOff+2] = _normals[_triangles[idxs[i]].nIdx[j]].Z();
				}

				if (hasTexCoords)
				{
					tmpBuf[k*numComponents+texOff] = _texcoords[_triangles[idxs[i]].tIdx[j]].X();
					tmpBuf[k*numComponents+texOff+1] = _texcoords[_triangles[idxs[i]].tIdx[j]].Y();
				}

				k++;

			}
			else
			{
				nElem[i*3+j] = mapping[id];
			}
		}
		
	}
	/*for(int i=0; i<k; i++)
	{		
		printf("mid: %f  oid:%f vertex: %f %f %f normal: %f %f %f\n",
			tmpBuf[i*numComponents],
			tmpBuf[i*numComponents+1],
			tmpBuf[i*numComponents+2],
			tmpBuf[i*numComponents+3],
			tmpBuf[i*numComponents+4],
			tmpBuf[i*numComponents+5],
			tmpBuf[i*numComponents+6],
			tmpBuf[i*numComponents+7]);
	}*/
	/*for(int i=0; i< idxs.size(); i++)
	{
		printf("%d %d %d\n",nElem[i*3],nElem[i*3+1],nElem[i*3+2]);
	}*/
	VAO->SetVertexArray(k*sizeof(float)*numComponents, &tmpBuf[0]);
	VAO->SetElementArray(GL_UNSIGNED_INT,idxs.size()*3*sizeof( unsigned int ),nElem);
	VAO->Bind();
	VAO->EnableAttribute( IGLU_ATTRIB_VERTEX, 3, GL_FLOAT, vertStride, BUFFER_OFFSET(vertOff*sizeof(float)));
	if (hasNormals)  VAO->EnableAttribute( IGLU_ATTRIB_NORMAL, 
		                                        3, GL_FLOAT, vertStride, BUFFER_OFFSET(normOff*sizeof(float)));
	if (hasTexCoords)   VAO->EnableAttribute( IGLU_ATTRIB_TEXCOORD, 
		                                        2, GL_FLOAT, vertStride, BUFFER_OFFSET(texOff*sizeof(float)));
	if (reader->HasMatlID())  VAO->EnableAttribute( IGLU_ATTRIB_MATL_ID, 
		                                        1, GL_FLOAT, vertStride, BUFFER_OFFSET(matlIdOff*sizeof(float)));
	
	VAO->Unbind();
	//free(tmpBuf);
	free(nElem);
	

}
void VFCIGLUApp::FrustumCulling(Frustum& frustum,vector<BBox>&aabbs, vector<int>& passedIdx)
{
	int total = aabbs.size();
	//对AABB进行裁剪，把通过的id保存下来
	/*vector<int> passedIdx;
	passedIdx.reserve(total);*/
	passedIdx.clear();

	for(int i=0; i<total; i++)
	{
		if (frustum.ContainsBBox(aabbs[i]) != Out)
		{
			passedIdx.push_back(i);
		}
	}

	printf("total %d triangles, after frustum cull remain %d triangles\n",total,passedIdx.size());
}
void VFCIGLUApp::FrustumCulling(Frustum& frustum,vector<AaBox>&aabbs, vector<int>& passedIdx)
{
	int total = aabbs.size();
	//对AABB进行裁剪，把通过的id保存下来
	/*vector<int> passedIdx;
	passedIdx.reserve(total);*/
	passedIdx.clear();

	for(int i=0; i<total; i++)
	{
		if (frustum.ContainsAaBox(aabbs[i]) != Out)
		{
			passedIdx.push_back(i);
		}
	}
#ifdef _DEBUG
	printf("total %d triangles, after frustum cull remain %d triangles\n",total,passedIdx.size());
#endif
	
}

void VFCIGLUApp::InitScene()
{
	for (int i=0; i<_sceneData->getObjects().size(); i++)
	{
		SceneObject* obj = _sceneData->getObjects()[i];
		switch (obj->getType())
		{
		case SceneObjType::mesh:
			{
				ObjModelObject* mesh = (ObjModelObject*)obj;
				
				IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(),IGLU_OBJ_UNITIZE);
				_objReaders.push_back(objReader);
				glm::mat4 trans = mesh->getTransform();						
				_objTransforms.push_back(IGLUMatrix4x4(&trans[0][0]));
				GetAABBs(objReader);
				
				break;
			}
		default:
			break;
		}
	}

	// We've loaded all our materials, so prepare to use them in rendering
	IGLUOBJMaterialReader::FinalizeMaterialsForRendering(IGLU_TEXTURE_REPEAT);
	
	
}

void VFCIGLUApp::Display()
{

	// Start timing this frame draw
	_frameRate->StartFrame();
	glDisable(GL_BLEND);
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	for(int i=0; i<_objReaders.size(); i++)
	{
		Frustum frustum(_camera->GetProjectionMatrix(),   _camera->GetViewMatrix() * _objTransforms[i]);
		vector<int> idxs;
		_bvh->frustumCulling(frustum,idxs);
		if (idxs.size() ==0)
			continue;
		UpdateReader(idxs,_objReaders[i]);
		_shaders[0]->Enable();
		_shaders[0]["project"]         = _camera->GetProjectionMatrix();
		_shaders[0]["model"]           = _objTransforms[i];
		_shaders[0]["view"]            = _camera->GetViewMatrix();
		_shaders[0]["lightIntensity" ] = float(1);
		_shaders[0]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[0]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		//_shaders[0]["lightPos"] = _lightPos;
		_shaders[0]["lightColor"] = _lightColor;
		
		//FrustumCulling(frustum,_aabbs,idxs);
		//FrustumCulling(frustum,_bboxs,idxs);
#ifdef _DEBUG
		printf("total %d triangles, after frustum cull remain %d triangles\n",_bvh->NumOfPrims(),idxs.size());
#endif
		//_vao = new IGLUVertexArray();
		//SetupNewVAO(_vao,idxs,_objReaders[i]);
		//_objReaders[i]->Draw(_shaders[0]);
	
		
		//_vao->DrawElements( GL_TRIANGLES, 3* idxs.size() );
	    _objReaders[i]->Draw(_shaders[0]);

		_shaders[0]->Disable();
		
		//delete _vao;
	}



	

	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}