#ifndef GIMAPP_H
#define GIMAPP_H
#include "Frustum.h"
#include "BBoxVector.hpp"
#include "Triangle.h"
#include "TriangleVector.hpp"
#include "BVH.h"
#include "IGLUApp.h"
struct InstanceData
{
	InstanceData():position(vec4(0,0,0,1)),normal(vec4(0.f)){}
	vec4 position;
	vec4 normal;
};
class GIMApp : public IGLUApp
{
public:
	GIMApp(const char* sceneFile):IGLUApp(sceneFile)
	{
		_cpuTimer = new IGLUCPUTimer();
		_gpuTimer = new IGLUGPUTimer();
	};
	virtual ~GIMApp();
	virtual void InitScene();
	virtual void InitShaders();
	virtual void Display();
	void DisplayOld();
	void InitShadersOld();
	void InitBuffer();
	void InitBufferOld();
	//更新vao的element array，只绘制正面的镜面
	int UpdateMirrorVAO();
	void GetAABBs(IGLUOBJReader::Ptr &reader, IGLUMatrix4x4& model);
	//只需要更新reader的elements array
	void UpdateReader( vector<int>& idxs, IGLUOBJReader::Ptr &reader);
	//根据镜面三角形(p1,p2,p3)创建虚视锥vfrustum;
	void CreateVirtualFrustum(vec3& p1, vec3& p2, vec3& p3, Frustum& vfrustum);
	//根据镜面p,normal创建虚视锥vfrustum;
	void CreateVirtualFrustum(vec3& p, vec3& normal, Frustum& vfrustum);
	void DisplayFrustum(Frustum& frustum);
private:
	IGLUShaderProgram::Ptr _mirrorShader;
	IGLUShaderProgram::Ptr _giShader;
	IGLUShaderProgram::Ptr _objShader;
	IGLUShaderProgram::Ptr _mirrorTexShader;
	IGLUShaderProgram::Ptr _testShader;
	IGLUShaderProgram::Ptr _simpleShader;
	//把镜面数据保存在buffer中
	IGLUBuffer::Ptr _mirrorDataBuffer;
	IGLUTextureBuffer::Ptr _mirrorBuffer;
	IGLUBuffer::Ptr _mirrorUB;
	//镜面在场景mesh列表中的id	
	int _mirrorId;
	//蒙版绘制时的镜面VAO
	IGLUVertexArray::Ptr _mirrorVAO;
	IGLUFramebuffer::Ptr _mirrorStencilFBO;
	IGLUFramebuffer::Ptr _mirrorFBO;
	IGLUMatrix4x4 _mirrorTransform;
	//保存镜面数据的buffer（顶点）	
	vector<InstanceData> _instanceDatum;
	float* _VAOBuffer;
	GLuint _SSBO;
	GLuint _counterBuffer;
	
	IGLUCPUTimer::Ptr _cpuTimer;
	IGLUGPUTimer::Ptr _gpuTimer;
	//virtual view frustum;
	vector<Frustum> _vFrustums;
	
	vector<BBox> _bboxs;	
	vector<Object*> _triangleObjects;	
	BVH * _bvh;
};



#endif