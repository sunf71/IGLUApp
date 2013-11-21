#ifndef GIMAPP_H
#define GIMAPP_H

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
	//����vao��element array��ֻ��������ľ���
	int UpdateMirrorVAO();
private:
	IGLUShaderProgram::Ptr _mirrorShader;
	IGLUShaderProgram::Ptr _giShader;
	IGLUShaderProgram::Ptr _objShader;
	IGLUShaderProgram::Ptr _mirrorTexShader;
	IGLUShaderProgram::Ptr _testShader;
	//�Ѿ������ݱ�����buffer��
	IGLUBuffer::Ptr _mirrorDataBuffer;
	IGLUTextureBuffer::Ptr _mirrorBuffer;
	IGLUBuffer::Ptr _mirrorUB;
	//�����ڳ���mesh�б��е�id	
	int _mirrorId;
	//�ɰ����ʱ�ľ���VAO
	IGLUVertexArray::Ptr _mirrorVAO;
	IGLUFramebuffer::Ptr _mirrorStencilFBO;
	IGLUFramebuffer::Ptr _mirrorFBO;
	IGLUMatrix4x4 _mirrorTransform;
	//���澵�����ݵ�buffer�����㣩	
	vector<InstanceData> _instanceDatum;
	float* _VAOBuffer;
	GLuint _SSBO;
	GLuint _counterBuffer;
	
	IGLUCPUTimer::Ptr _cpuTimer;
	IGLUGPUTimer::Ptr _gpuTimer;
};



#endif