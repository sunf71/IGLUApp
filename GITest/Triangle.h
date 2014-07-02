#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "iglu.h"
#include "Object.h"
using namespace iglu;
class Triangle : public Object
{
private:
	int _id;	
	vec3 _center;
public:
	Triangle(){}
	vec3 p1,p2,p3;
	Triangle(const vec3& v1, const vec3& v2, const vec3& v3,int id)
	{
		_id = id;
		p1 = vec3(v1);
		p2 = vec3(v2);
		p3 = vec3(v3);
		vec3 center = (v1 + v2 + v3)/3;
		_center = vec3(center.X(),center.Y(),center.Z());
		
	}
	Triangle(const vec3& v1, const vec3& v2, const vec3& v3):p1(v1),p2(v2),p3(v3)
	{
		vec3 center = (v1 + v2 + v3)/3;
		_center = vec3(center.X(),center.Y(),center.Z());
	}

	virtual bool getIntersection(
		const Ray& ray, 
		IntersectionInfo* intersection)  const
	{
		//not implemented
		return true;
	}

	//! Return an object normal based on an intersection
	virtual vec3 getNormal(const IntersectionInfo& I) const
	{
		return vec3((p3-p2).Cross(p1-p2));
	}

	//! Return a bounding box for this object
	virtual BBox getBBox() const
	{ 
	/*	BBox _bbox(p1);
		_bbox.expandToInclude(p2);
		_bbox.expandToInclude(p3);*/
		float minX,maxX,minY,maxY,minZ,maxZ;
		if (p1.X() < p2.X())
		{
			minX = p1.X();
			maxX = p2.X();
		}
		else
		{
			minX = p2.X();
			maxX = p1.X();
		}
		if (p1.Y() < p2.Y())
		{
			minY = p1.Y();
			maxY = p2.Y();
		}
		else
		{
			minY = p2.Y();
			maxY = p1.Y();
		}
		if (p1.Z() < p2.Z())
		{
			minZ = p1.Z();
			maxZ = p2.Z();
		}
		else
		{
			minZ = p2.Z();
			maxZ = p1.Z();
		}
		if (p3.X() < minX)
			minX = p3.X();
		if(p3.Y() < minY)
			minY = p3.Y();
		if(p3.Z() < minZ)
			minZ = p3.Z();
		if(p3.X() > maxX)
			maxX = p3.X();
		if(p3.Y() > maxY)
			maxY = p3.Y();
		if(p3.Z() >maxZ)
			maxZ = p3.Z();
		return BBox(vec3(minX,minY,minZ),vec3(maxX,maxY,maxZ));
	}

	//! Return the centroid for this object. (Used in BVH Sorting)
	virtual vec3 getCentroid() const
	{
		return _center;
	}

	int getId() const
	{
		return _id;
	}
};

//用于创建BVH
class Box  : public Object
{
private:
	int _id;	
	vec3 _center;
public:
	Box(){}
	vec3 min,max;
	Box(const vec3& m, const vec3& M, int id):_id(id)
	{
		min = m;
		max = M;
		vec3 center = (m+M)/2;
		_center = vec3(center.X(),center.Y(),center.Z());
		
	}
	
	virtual bool getIntersection(
		const Ray& ray, 
		IntersectionInfo* intersection)  const
	{
		//not implemented
		return true;
	}

	//! Return an object normal based on an intersection
	virtual vec3 getNormal(const IntersectionInfo& I) const
	{
		return vec3(0,0,0);
	}

	//! Return a bounding box for this object
	virtual BBox getBBox() const
	{ 
	/*	BBox _bbox(p1);
		_bbox.expandToInclude(p2);
		_bbox.expandToInclude(p3);*/

		return BBox(min,max);
	}

	//! Return the centroid for this object. (Used in BVH Sorting)
	virtual vec3 getCentroid() const
	{
		return _center;
	}

	int getId() const
	{
		return _id;
	}
};
#endif