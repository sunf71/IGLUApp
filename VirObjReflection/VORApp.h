#ifndef VORAPP_H
#define VORAPP_H
#include "iglu.h"
#include "IGLUApp.h"
#include "BVH/frustum.h"
namespace OGL
{
	using namespace iglu;
	class VORApp:public IGLUApp
	{
	public:
		VORApp(const char* fileName):IGLUApp(fileName)
		{}
		~VORApp()
		{
			
		}
		void VirtualFrustumsCulling();
		virtual void InitScene();

		virtual void InitShaders();

		virtual void Display();
	private:
		std::vector<IGLUOBJReader::Ptr> _mirrorObjs;
		std::vector<IGLUMatrix4x4> _mirrorTransforms;
		//三角形总数量
		size_t _triSize;
	};
}

#endif