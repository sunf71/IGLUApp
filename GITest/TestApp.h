#ifndef TESTAPP_H
#define TESTAPP_H

#include "IGLUApp.h"
using namespace iglu;

class TestApp : public IGLUApp
{
public:
	TestApp(const char* sceneFile):IGLUApp(sceneFile){};
	virtual ~TestApp();
	virtual void InitScene();
	virtual void InitShaders();
	virtual void Display();
	void InitBuffer();

private:
	IGLUShaderProgram::Ptr _objShader;
	//�Ѿ������ݱ�����buffer��
	IGLUBuffer::Ptr _mirrorDataBuffer;
	IGLUTextureBuffer::Ptr _mirrorBuffer;
	GLuint _SSBO;
};

#endif