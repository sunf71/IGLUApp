#include "VirtualFrustumTest.h"
#include<glm\glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <time.h>
#include "IGLUMathHelper.h"

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

void VirtualFrustumApp::Display()
{
	// Start timing this frame draw
	_frameRate->StartFrame();
	glDisable(GL_BLEND);
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	IGLUShaderProgram::Ptr shader = _shaders[1];
	
	for(int i=0; i<_objReaders.size(); i++)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 
		shader->Enable();
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
	//DisplayFrustum(*_vFrustum);
	_vFrustum->Draw();
	glPolygonMode(GL_FRONT, GL_FILL); 
	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}