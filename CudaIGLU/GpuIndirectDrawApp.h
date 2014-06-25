#pragma once
#include "igluApp.h"
#include "cuda_gl_interop.h"
#include "cuda_runtime_api.h"
namespace OGL
{
	using namespace iglu;
#define NUM_DRAWS 64*64
	struct InstanceData
	{
		InstanceData():position(vec4(0,0,0,1)),normal(vec4(0.f)){}
		vec4 position;
		vec4 normal;
	};
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

	class GpuIndirectDrawApp:public IGLUApp
	{
	public:
		GpuIndirectDrawApp(const char* fileName):IGLUApp(fileName)
		{
			_dtTime = 0;
		}
		~GpuIndirectDrawApp()
		{
			safe_delete(_instanceData);
			safe_delete( _grassBuffer);
			safe_delete ( _grassTexBuffer);
			safe_delete(_drawIndexBuffer);
			safe_delete(_indirectDrawBuffer);
			cudaGraphicsUnregisterResource( resource );
			
			
		}
		void InitCudaOGL();
		void InitInstanceData();
		void InitAttribute();
		void InitDrawArrayCommand();
		void InitDrawElementCommand();
		void UpdateDrawCommand();
		void SetUpdateDrawCommand(void(*u) (unsigned int*,unsigned int,unsigned int))
		{
			update = u;
		}
		virtual void InitScene();

		virtual void InitShaders();

		virtual void Display();

	private:
		InstanceData* _instanceData;
		IGLUBuffer::Ptr _grassBuffer;
		IGLUTextureBuffer::Ptr _grassTexBuffer;
		IGLUBuffer::Ptr _indirectDrawBuffer;		
		IGLUBuffer::Ptr _drawIndexBuffer;
		float _dtTime;
		cudaGraphicsResource *resource;
		void(*update) (unsigned int*,unsigned int,unsigned int); 
	};
}