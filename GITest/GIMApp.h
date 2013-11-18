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
	//更新vao的element array，只绘制正面的镜面
	int UpdateMirrorVAO();
private:
	IGLUShaderProgram::Ptr _mirrorShader;
	IGLUShaderProgram::Ptr _giShader;
	IGLUShaderProgram::Ptr _objShader;
	IGLUShaderProgram::Ptr _mirrorTexShader;
	//把镜面数据保存在buffer中
	IGLUTextureBuffer::Ptr _mirrorBuffer;
	//镜面在场景mesh列表中的id	
	int _mirrorId;
	//蒙版绘制时的镜面VAO
	IGLUVertexArray::Ptr _mirrorVAO;
	IGLUFramebuffer::Ptr _mirrorStencilFBO;
	IGLUFramebuffer::Ptr _mirrorFBO;
	IGLUMatrix4x4 _mirrorTransform;
	//保存镜面数据的buffer（顶点）
	
	vector<vec3> _mirrorNormals;
	GLuint _SSBO;
	GLuint _counterBuffer;
};



#endif