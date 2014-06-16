#include "VirtualFrustumTest.h"
#include<glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <time.h>
#include "IGLUMathHelper.h"
#ifdef far
#undef far
#endif
void VirtualFrustumApp::InitShaders()
{
	_objShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/object.vert.glsl","../../CommonSampleFiles/shaders/object.frag.glsl");
	_objShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 
	_simpleShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/simple.vert.glsl","../../CommonSampleFiles/shaders/simple.frag.glsl");

	_shaders.push_back(_objShader);
	_shaders.push_back(_simpleShader);
}

void VirtualFrustumApp::CreateVirtualFrustum(vec3& p1, vec3& p2, vec3& p3, Frustum& vfrustum)
{
	vec3 _eye = _camera->GetEye();
	vec3 _at = _camera->GetAt();

	vec3 normal = ((p3-p2).Cross(p1-p2));
	normal.Normalize();
	vec3 vEye = _eye - 2*(_eye-p1).Dot(normal)*normal;
	vec3 vAt;	
	vec3 dir = (_eye-vEye);
	dir.Normalize();
	if (!IGLUMathHelper::LineIntersecPlane(vEye,dir,p1,p2,p3,vAt))
	{
		printf("Error in CreateVirtual Frustum\n");
		return;
	}
	vec3 up = _camera->GetUp();
	vec3 right = dir.Cross(up);
	right.Normalize();
	vec3 y;
	y[0] = abs((p1-vAt).Dot(up));
	y[1] = abs((p2-vAt).Dot(up));
	y[2] = abs((p3-vAt).Dot(up));
	float yMax = y.MaxComponent();
	vec3 x;
	x[0] = abs((p1-vAt).Dot(right));
	x[1] = abs((p2-vAt).Dot(right));
	x[2] = abs((p3-vAt).Dot(right));
	float xMax = x.MaxComponent();
	float nearZ = (vAt-vEye).Length();
	float fovY =  2*atan(yMax/nearZ)/IGLU_PI*180;
	float ar = yMax / xMax;
	/*farZ = 20;*/
	vfrustum = Frustum(vEye,vAt,up,fovY,nearZ,_camera->GetFar(),ar);
}
void VirtualFrustumApp::InitScene()
{
	IGLUApp::InitScene();
	GenVirtualFrustum();
}
void VirtualFrustumApp::CreateVirtualFrustum(vec3& p1, vec3& normal, Frustum& vfrustum)
{
	vec3 _eye = _camera->GetEye();
	vec3 _at = _camera->GetAt();
	
	vec3 vEye = _eye - 2*(_eye-p1).Dot(normal)*normal;
	vec3 vAt = _at - 2*(_at-p1).Dot(normal)*normal;
	vec3 inter;
	vec3 dir = (vAt-vEye);
	dir.Normalize();
	if (!IGLUMathHelper::LineIntersecPlane(vEye,dir,p1,normal,inter))
	{
		printf("Error in CreateVirtual Frustum\n");
		return;
	}
	float nearZ = (vEye - inter).Length();
	vfrustum = Frustum(vEye,vAt,_camera->GetUp(),_camera->GetFovY(),_camera->GetNear(),_camera->GetFar(),_camera->GetAspectRatio());
}
void VirtualFrustumApp::DisplayFrustum(Frustum& frustum)
{
	
	_simpleShader->Enable();
	_simpleShader["mvp"] = _camera->GetProjectionMatrix()* _camera->GetViewMatrix();
	frustum.Draw();
	_simpleShader->Disable();
}

bool VirtualEye(const vec3& eye, const vec3& p1, const vec3& p2, const vec3& p3, vec3& vEye)
{
	vec3 normal = (p3-p2).Cross(p1-p2);
	normal.Normalize();
	float dir = (eye-p1).Dot(normal);
	if ( dir > 0)
	{
		vEye = eye - 2*dir*normal;
		return true;
	}
	else
		return false;

}
void VirtualFrustumApp::GenVirtualFrustum()
{
	//ÐéÊÓµã
	vec3 eye(-10,0,0);
	vec3 virEye;
	float far = 100;
	IGLUOBJReader::Ptr reader = _objReaders[0];
	int total = reader->GetTriangleCount();
	_tFrustumVec.reserve(total);
	vector<IGLUOBJTri *> tris = reader->GetTriangles();
	vector<vec3> vertices = reader->GetVertecies();
	for(int i=0; i<total; i++)
	{
		IGLUOBJTri* tri = tris[i];
		
		Triangle triangle( vertices[tri->vIdx[0]],vertices[tri->vIdx[1]],vertices[tri->vIdx[2]]);
		if (VirtualEye(eye,triangle.p1,triangle.p2,triangle.p3,virEye))
		{
			_tFrustumVec.push_back(new TriFrustum(virEye,triangle,far));
		}
		
	}
		
}
void  CullingObjReader(TriFrustum* frustum, IGLUOBJReader::Ptr reader, vector<int>& idx)
{
	size_t size = reader->GetTriangles().size();

	idx.reserve(size);
	for(int i=0; i<size; i++)
	{
		vector<vec3>& vertices = reader->GetVertecies();
		Triangle triangle(vertices[reader->GetTriangles()[i]->vIdx[0]],
			vertices[reader->GetTriangles()[i]->vIdx[1]],
			vertices[reader->GetTriangles()[i]->vIdx[2]]);
		if ( frustum->ContainsTriangle(&triangle) != Out )
		{
			idx.push_back(i);
		}
	}

}
void VirtualFrustumApp::Display()
{
	// Start timing this frame draw
	_frameRate->StartFrame();
	glDisable(GL_BLEND);
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	IGLUShaderProgram::Ptr shader = _shaders[1];
	
	for (int j=0; j< _tFrustumVec.size(); j++)
	{
		vector<int>idx;
		for(int i=1; i<_objReaders.size(); i++)
		{
			CullingObjReader(_tFrustumVec[0],_objReaders[i],idx);
			UpdateReader(idx,_objReaders[i]);
			shader->Enable();
			shader["vcolor"] = vec4(0,0,1,0);
			shader["mvp"] = _camera->GetProjectionMatrix()*_camera->GetViewMatrix()*_objTransforms[i];
			_objReaders[i]->Draw(shader);
			shader->Disable();
			//_shaders[0]->Enable();
			//_shaders[0]["project"]         = _camera->GetProjectionMatrix();
			//_shaders[0]["model"]           = _objTransforms[i];
			//_shaders[0]["view"]            = _camera->GetViewMatrix();
			//_shaders[0]["lightIntensity" ] = float(1);
			//_shaders[0]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
			//_shaders[0]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
			////_shaders[0]["lightPos"] = _lightPos;
			//_shaders[0]["lightColor"] = _lightColor;	
			//_objReaders[i]->Draw(_shaders[0]);
			//_shaders[0]->Disable();

		}
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 
		//_vFrustum->Draw();
		_tFrustumVec[j]->Draw();
		glPolygonMode(GL_FRONT, GL_FILL); 
	}
	
	
	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}

void VirtualFrustumApp::UpdateReader( vector<int>& idxs, IGLUOBJReader::Ptr &reader)
{
	IGLUVertexArray::Ptr vao = reader->GetVertexArray();
	uint* elementArray = reader->GetElementArrayData();
	vector<uint> indices;
	indices.reserve(idxs.size());
	for(int i=0; i<idxs.size(); i++)
	{
		for(int j=0; j<3; j++)
		{
			indices.push_back(elementArray[idxs[i]*3+j]);
		}
	}
	vao->SetElementArray(GL_UNSIGNED_INT,sizeof(uint)*indices.size(),&indices[0]);
}