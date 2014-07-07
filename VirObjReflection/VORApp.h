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
		//镜面三角形顶点数组，每个顶点用3个浮点型，每个三角形3个顶点
		std::vector<float> _mirrorPos;
		//每个instance包含镜面上一点vec4和镜面法线vec4
		std::vector<float> _instanceData;
		//三角形总数量
		size_t _triSize;
		cudaGraphicsResource *_resource, *_elemSource;
		IGLUShaderProgram::Ptr _objShader;
		IGLUShaderProgram::Ptr _simpleShader;
		
	};
}

#endif