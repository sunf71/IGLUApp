#pragma once
#include "igluApp.h"
#include "GIMApp.h"
#define NUM_DRAWS 64*64

struct DrawArraysIndirectCommand
{
    GLuint  count;
    GLuint  primCount;
    GLuint  first;
    GLuint  baseInstance;
};
typedef  struct {
        uint  count;
        uint  instanceCount;
        uint  firstIndex;
        uint  baseVertex;
        uint  baseInstance;
    } DrawElementsIndirectCommand;

class IndirectDrawApp:public IGLUApp
{
public:
	IndirectDrawApp(const char* fileName):IGLUApp(fileName)
	{
		_dtTime = 0;
	}
	~IndirectDrawApp()
	{
		safe_delete(_instanceData);
		safe_delete(_grassBuffer);
		safe_delete(_grassTexBuffer);

	}
	void InitInstanceData();
	void InitDrawArrayCommand();
	void InitDrawElementCommand();
	virtual void InitScene();

	virtual void InitShaders();

	virtual void Display();

private:
	InstanceData* _instanceData;
	IGLUBuffer::Ptr _grassBuffer;
	IGLUTextureBuffer::Ptr _grassTexBuffer;
	GLuint  indirect_draw_buffer;
    GLuint  draw_index_buffer;
	float _dtTime;
};