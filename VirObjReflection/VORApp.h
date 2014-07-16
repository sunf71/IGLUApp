#ifndef VORAPP_H
#define VORAPP_H
#include "iglu.h"
#include "IGLUApp.h"
#include "BVH/frustum.h"
#include "cuda_gl_interop.h"
#include "cuda_runtime_api.h"
#include "CudaFunctions.h"
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
		{
			_cpuTimer = new IGLUCPUTimer();
			_gpuTimer = new IGLUGPUTimer();
		}
		~VORApp()
		{
			safe_delete(_attribBuffer);
			safe_delete(_indirectDrawBuffer);
			safe_delete(_cpuTimer);
			safe_delete(_gpuTimer);
			_mirrorObjs.clear();
			cudaGraphicsUnregisterResource( _attriRes );
			cudaGraphicsUnregisterResource(_cmdRes);
			cuda::ReleaseGPUMemory();
		}
		void InitBuffer();
		void InitOGLCuda();
		void InitMirrorData();
		void InitAttribute();
		size_t VirtualFrustumsCulling(size_t& frustumSize);
		size_t CPUVirtualFrustumsCulling(size_t& frustumSize);
		void UpdateVirtualObject(size_t size);
		virtual void InitScene();

		virtual void InitShaders();

		virtual void Display();
	private:
		//�������������
		IGLUBuffer::Ptr _indirectDrawBuffer;		
		//�����徵��������
		IGLUBuffer::Ptr _attribBuffer;
		//���涥��ͷ�������
		IGLUBuffer::Ptr _mirrorBuffer;
		//���������б�
		IGLUBuffer::Ptr _mirrorElementBuffer;
		IGLUTextureBuffer::Ptr _instanceDataTex;
		std::vector<IGLUOBJReader::Ptr> _mirrorObjs;
		std::vector<IGLUMatrix4x4> _mirrorTransforms;
		std::vector<unsigned> _mirrorTransId;
		//���������ζ������飬ÿ��������3�������ͣ�ÿ��������3������
		std::vector<float> _mirrorPos;
		//ÿ��instance����������һ��vec4�;��淨��vec4
		std::vector<float> _instanceData;
		//�Ǿ���������������
		size_t _triSize;
		cudaGraphicsResource *_attriRes, *_cmdRes, *_mCEleRes, *_mEleRes;
		IGLUShaderProgram::Ptr _objShader;
		IGLUShaderProgram::Ptr _simpleShader;
		IGLUShaderProgram::Ptr _mirrorTexShader;

		IGLUFramebuffer::Ptr _mirrorStencilFBO;
		IGLUFramebuffer::Ptr _mirrorFBO;

		IGLUCPUTimer::Ptr _cpuTimer;
		IGLUGPUTimer::Ptr _gpuTimer;
	};
}

#endif