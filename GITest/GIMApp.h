#ifndef GIMAPP_H
#define GIMAPP_H

#include "IGLUApp.h"

class GIMApp : public IGLUApp
{
public:
	GIMApp(const char* sceneFile):IGLUApp(sceneFile){};
	virtual ~GIMApp();
	virtual void InitScene();
	virtual void InitShaders();
	virtual void Display();
	void InitBuffer();
	//����vao��element array��ֻ��������ľ���
	int UpdateMirrorVAO();
private:
	IGLUShaderProgram::Ptr _mirrorShader;
	IGLUShaderProgram::Ptr _giShader;
	IGLUShaderProgram::Ptr _objShader;
	IGLUShaderProgram::Ptr _mirrorTexShader;
	//�Ѿ������ݱ�����buffer��
	IGLUTextureBuffer::Ptr _mirrorBuffer;
	//�����ڳ���mesh�б��е�id	
	int _mirrorId;
	//�ɰ����ʱ�ľ���VAO
	IGLUVertexArray::Ptr _mirrorVAO;
	IGLUFramebuffer::Ptr _mirrorStencilFBO;
	IGLUFramebuffer::Ptr _mirrorFBO;
	IGLUMatrix4x4 _mirrorTransform;
	//���澵�����ݵ�buffer�����㣩
	
	vector<vec3> _mirrorNormals;
	GLuint _SSBO;
	GLuint _counterBuffer;
};



#endif