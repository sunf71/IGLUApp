#ifndef SCENEHELPER_H
#define SCENEHELPER_H

#include<glm\glm.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

enum SceneObjType
{
	group = 0,
	mesh,//obj文件载入
	sphere,//球
	parallelogram
};

//场景对象
class SceneObject
{
public:
	SceneObject(const std::string& name, const glm::mat4& transform):_objName(name),_transform(transform),_type(SceneObjType::group)
	{}
	
	SceneObject( ):_objName(""),_transform(glm::mat4(1.0)){}
	void setName(const std::string& name)
	{
		_objName = name;
	}
	std::string getName()
	{
		return _objName;
	}
	void setTransform(glm::mat4& trans)
	{
		_transform = trans;
	}
	glm::mat4 getTransform()
	{
		return _transform;
	}
	static void printMatrix(const glm::mat4& mat)
	{
		for(int i=0; i<4; i++)			
				std::cout<<mat[i][0]<<" "<<mat[i][1]<<" "<<mat[i][2]<<" "<<mat[i][3]<<std::endl;
		std::cout<<std::endl;
	}
	virtual void print()
	{
		std::cout<<"Object Name: "<<_objName<<" transform "<<std::endl;
		printMatrix(_transform);
	}
	SceneObjType getType()
	{
		return _type;
	}
	const std::string& getMaterialName()
	{
		return _materialName;
	}

protected:
	std::string _objName;
	std::string _materialName;
	glm::mat4 _transform;
	SceneObjType _type;
};
typedef std::vector<SceneObject *> SceneObjects;
typedef std::vector<SceneObject* >::iterator SceneObjItr;
class GroupObject : public SceneObject
{
public:
	GroupObject(const std::string& name, const glm::mat4& trans):SceneObject(name, trans)
	{}
	//GroupObject(char* ptr, FILE* f);
	~GroupObject()
	{
		for(int i=0; i<_groupObjs.size(); i++)
		{
			delete _groupObjs[i];
		}
		_groupObjs.clear();
	}
	void addObject(SceneObject* obj)
	{
		_groupObjs.push_back(obj);
	}
	SceneObjItr& getBeginItr()
	{
		return _groupObjs.begin();
	}
	SceneObjItr& getEndItr()
	{
		return _groupObjs.end();
	}
	virtual void print()
	{
		for(SceneObjItr  itr = _groupObjs.begin();
			itr != _groupObjs.end();
			++itr)
		{
			(*itr)->print();
		}
	}
private:
	SceneObjects _groupObjs;
};


class ObjModelObject : public SceneObject
{
public:
	ObjModelObject(const std::string& objName, const std::string& objFileName, const glm::mat4 trans)
		:_objFileName(objFileName),SceneObject(objName,trans)
	{
		_type = SceneObjType::mesh;
	}
	ObjModelObject(char* name, FILE* f);

public:
	std::string getObjFileName()
	{
		return _objFileName;
	}
private:
	std::string _objFileName;
};

class PlaneObject : public SceneObject
{
public:
	PlaneObject(const std::string& objName, const glm::mat4& transform,
		glm::vec3& anchor, glm::vec3& offset1, glm::vec3& offset2):_anchor(anchor),
		_v1(offset1),_v2(offset2),SceneObject(objName,transform)
	{
		_type = SceneObjType::parallelogram;
	}
	PlaneObject(char* name, FILE* f);	
public:
	glm::vec3& getAnchor()
	{
		return _anchor;
	}
	glm::vec3& getV1()
	{
		return _v1;
	}
	glm::vec3& getV2()
	{
		return _v2;
	}
private:
	glm::vec3 _anchor;
	glm::vec3 _v1;
	glm::vec3 _v2;
};
class CameraData
{
public:
	CameraData(FILE* f);
	CameraData():pinhole(true){}
	bool pinhole;
	glm::vec3 eye;
	glm::vec3 at;
	glm::vec3 up;
	float fovy;
	float zNear;
	float zFar;
	float resX,resY;
};
class Light
{
public:
	Light(FILE* f);
	glm::vec3 pos;
	glm::vec3 at;
	glm::vec3 color;
	int castShadow;
};
typedef std::vector<Light> Lights;


//scene data
class SceneData
{
public:
	SceneData(const char* file);
	~SceneData()
	{
		_lights.clear();
		for(int i=0; i<_objs.size(); i++)
			delete _objs[i];
		_objs.clear();
	}
	//打印场景测试
	void printScene()
	{
		SceneObjItr itr = getBeginItr();
		while(itr !=  getEndItr())
		{
			cout<<(*itr)->getName()<<endl;
			++itr;
		}
	}
	Lights& getLights()
	{
		return _lights;
	}
	CameraData& getCameraData()
	{
		return _camera;
	}
	SceneObjects& getObjects()
	{
		return _objs;
	}
	void loadCamera(char* ptr, FILE* file);
	void loadLights(char* ptr, FILE* file);
	void loadObjModel(char* ptr, FILE*file);

	SceneObjItr getBeginItr()
	{
		return _objs.begin();
	}
	SceneObjItr getEndItr()
	{
		return _objs.end();
	}
private:
	SceneObjects _objs;
	CameraData _camera;
	Lights _lights;
};

#endif