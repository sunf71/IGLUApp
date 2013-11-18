#ifndef VFCIGLUAPP_H
#define VFCIGLUAPP_H
#include "Frustum.h"
#include <stdio.h>
// All headers are automatically included from "iglu.h"
#include "iglu.h"
// IGLU classes and constants are all inside the iglu namespace.
#include <string>
#include <vector>
#include "IGLUApp.h"
#include "BBoxVector.hpp"
#include "Triangle.h"
#include "TriangleVector.hpp"
#include "BVH.h"
using namespace iglu;



/*
进行视锥体剪裁的App
*/

class VFCIGLUApp : public IGLUApp
{
public:
	VFCIGLUApp(const char* sceneFile):IGLUApp(sceneFile){};

	virtual ~VFCIGLUApp()
	{
		_aabbs.clear();
		_bboxs.clear();
		_normals.clear();
		_vertices.clear();
		_texcoords.clear();
		_triangles.clear();
		_triangleObjects.clear();
		if (_bvh)
		{
			delete _bvh;
			_bvh = NULL;
		}
	}
	 VFCIGLUApp(IGLUApp const&);              // Don't Implement
     void operator=(VFCIGLUApp const&); // Don't implement
public :
	
	void FrustumCulling(Frustum& frustum,vector<AaBox>&aabbs, vector<int>& passedIdx);
	void FrustumCulling(Frustum& frustum,vector<BBox>&aabbs, vector<int>& passedIdx);
	
	virtual void Display();
	
	virtual void InitScene();
	
	
	void MinMax(vec3& p1, vec3& p2, vec3& p3, int idx, float& min, float & max)
	{
		float p1v = p1.GetElement(idx);
		float p2v = p2.GetElement(idx);
		float p3v = p3.GetElement(idx);
		if ( p1v > p2v)
		{
			min= p2v;
			max = p1v;
		}
		else
		{
			min = p1v;
			max = p2v;
		}
		if (p3v<min)
		{
			min = p3v;
		}
		else if(p3v> max)
		{
			max = p3v;
		}
	}
	void GetAABB(vec3& p1,vec3& p2, vec3& p3, AaBox& aabox)
	{

		MinMax(p1,p2,p3,0,aabox.minX,aabox.maxX);
		MinMax(p1,p2,p3,1,aabox.minY,aabox.maxY);
		MinMax(p1,p2,p3,2,aabox.minZ,aabox.maxZ);

	}
	void GetAABB(vec3& p1, vec3& p2, vec3& p3, BBox& aabox)
	{
		aabox = BBox(Vector3(p1.X(),p1.Y(),p1.Z()));
		aabox.expandToInclude(Vector3(p2.X(),p2.Y(),p2.Z()));
		aabox.expandToInclude(Vector3(p3.X(),p3.Y(),p3.Z()));
	}
	void GetAABBs(IGLUOBJReader::Ptr & reader);

	void SetupNewVAO(IGLUVertexArray::Ptr& VAO, vector<int>& idxs, IGLUOBJReader::Ptr &reader);
protected:
	
	vector<AaBox> _aabbs;
	vector<BBox> _bboxs;
	vector<vec3> _normals;
	vector<vec3> _vertices;
	vector<vec2> _texcoords;
	vector<IGLUOBJTri> _triangles;
	vector<Object*> _triangleObjects;
	IGLUVertexArray::Ptr _vao;
	BVH * _bvh;
};

#endif