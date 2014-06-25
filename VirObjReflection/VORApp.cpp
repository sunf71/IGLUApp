#include "VORApp.h"
#include "CudaFunctions.h"
namespace OGL
{
	using namespace iglu;

	void VORApp::VirtualFrustumsCulling()
	{
		vec3 eye = GetCamera()->GetEye();
		cuda::VirtualFrustumCulling(_triSize,eye,150,&_mirrorObjs[0],&_mirrorTransforms[0],_mirrorObjs.size());
	}
	void VORApp::InitScene()
	{
		for (int i=0; i<_sceneData->getObjects().size(); i++)
		{
			SceneObject* obj = _sceneData->getObjects()[i];
			switch (obj->getType())
			{
			case SceneObjType::mesh:
				{
					ObjModelObject* mesh = (ObjModelObject*)obj;
					
					IGLUOBJReader::Ptr objReader;

					if (mesh->getUnitizeFlag())
					{
						objReader = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(), IGLU_OBJ_UNITIZE | IGLU_OBJ_COMPACT_STORAGE);
					}
					else
					{
						objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(),IGLU_OBJ_COMPACT_STORAGE);
					}
					
					glm::mat4 trans = (mesh->getTransform());					
					IGLUMatrix4x4 model(&trans[0][0]);
					
					if (!obj->getMaterialName().compare("mirror"))
					{
						_mirrorObjs.push_back(objReader);
						_mirrorTransforms.push_back(model);
					}
					else
					{
						_objReaders.push_back(objReader);
						_objTransforms.push_back(model);
					}
					//delete model;
					break;
				}
			default:
				break;
			}
		}
		// We've loaded all our materials, so prepare to use them in rendering
		IGLUOBJMaterialReader::FinalizeMaterialsForRendering(IGLU_TEXTURE_REPEAT);

		_triSize = cuda::BuildBvh(&_objReaders[0], &_objTransforms[0], _objReaders.size());
		
	}

	void VORApp::InitShaders()
	{
		IGLUApp::InitShaders();
	}

	void VORApp::Display()
	{
		VirtualFrustumsCulling();
		IGLUApp::Display();
	}
}