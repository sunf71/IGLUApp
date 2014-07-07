#ifndef VORAPP_H
#define VORAPP_H
#include "iglu.h"
#include "IGLUApp.h"
#include "BVH/frustum.h"
#include "cuda_gl_interop.h"
#include "cuda_runtime_api.h"
namespace OGL
{
	using namespace iglu;
	typedef  struct {
		uint  count;
		uint  instanceCount;
		uint  firstIndex;
		uint  baseVertex;
		uint  baseInstance;
	} DrawElementsIndirectCommand;
	class VORApp:public IGLUApp
	{
	public:
		VORApp(const char* fileName):IGLUApp(fileName)
		{}
		~VORApp()
		{
			safe_delete(_attribBuffer);
			safe_delete(_indirectDrawBuffer);
			_mirrorObjs.clear();
			cudaGraphicsUnregisterResource( _resource );
			cudaGraphicsUnregisterResource(_elemSource);
		}
		void InitBuffer();
		void InitOGLCuda();
		void InitMirrorData();
		void InitAttribute();
		size_t VirtualFrustumsCulling();
		void UpdateVirtualObject(size_t size);
		virtual void InitScene();

		virtual void InitShaders();

		virtual void Display();
	private:
		IGLUBuffer::Ptr _indirectDrawBuffer;		
		IGLUBuffer::Ptr _attribBuffer;
		IGLUBuffer::Ptr _mirrorBuffer;
		IGLUTextureBuffer::Ptr _instanceDataTex;
		std::vector<IGLUOBJReader::Ptr> _mirrorObjs;
		std::vector<IGLUMatrix4x4> _mirrorTransforms;
		std::vector<unsigned> _mirrorTransId;
		//���������ζ������飬ÿ��������3�������ͣ�ÿ��������3������
		std::vector<float> _mirrorPos;
		//ÿ��instance����������һ��vec4�;��淨��vec4
		std::vector<float> _instanceData;
		//������������
		size_t _triSize;
		cudaGraphicsResource *_resource, *_elemSource;
		IGLUShaderProgram::Ptr _objShader;
		IGLUShaderProgram::Ptr _simpleShader;
		
	};
}

#endif