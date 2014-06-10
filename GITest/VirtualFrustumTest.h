#ifndef VIRTUALFRUSTUM_H
#define VIRTUALFRUSTUM_H
#include "Frustum.h"
#include "BBoxVector.hpp"
#include "Triangle.h"
#include "TriangleVector.hpp"
#include "BVH.h"
#include "IGLUApp.h"
#include "Frustum.h"

class VirtualFrustumApp : public IGLUApp
{
public:
	VirtualFrustumApp(const char* cfgFile) : IGLUApp(cfgFile)
	{
		Camera* realCam = GetCamera();
		vec3 offset = vec3(0,0,-5);
		/*_vat = realCam->GetAt() + offset;
		_veye = realCam->GetEye() + offset;	   
		_vup = realCam->GetUp();
		_vnear = 3;
		_vfar = realCam->GetFar() - offset.Z();	
		_vfovy = realCam->GetFovY();*/
		_veye = vec3(-5,0,0);
		_vat = vec3(0,0,0);
		_vup = vec3(0,1,0);
		_vnear = 3;
		_vfar = 195;
		_vfovy = 60;
		_vFrustum = new Frustum(_veye,_vat,_vup,_vfovy,_vnear,_vfar,realCam->GetAspectRatio());
		CreateVirtualFrustum(vec3(0,0,0),vec3(0,1,0),vec3(1,1,0),*_vFrustum);
	
	};
	~VirtualFrustumApp()
	{
		safe_delete(_vFrustum);
	};
	virtual void InitShaders();
	virtual void Display();
	//根据镜面三角形(p1,p2,p3)创建虚视锥vfrustum;
	void CreateVirtualFrustum(vec3& p1, vec3& p2, vec3& p3, Frustum& vfrustum);
	//根据镜面p,normal创建虚视锥vfrustum;
	void CreateVirtualFrustum(vec3& p, vec3& normal, Frustum& vfrustum);
	void DisplayFrustum(Frustum& frustum);
private:
	IGLUShaderProgram::Ptr _objShader;
	IGLUShaderProgram::Ptr _simpleShader;
	//虚视点
	iglu::vec3 _vat;
	iglu::vec3 _veye;
	iglu::vec3 _vup;
	float _vfovy;
	float _vnear;
	float _vfar;
	Frustum* _vFrustum;
	
};
#endif