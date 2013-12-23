#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "iglu.h"
#include "Object.h"
using namespace iglu;
class Triangle : public Object
{
private:
	int _id;	
	Vector3 _center;
public:
	Triangle();
	Vector3 p1,p2,p3;
	Triangle(const vec3& v1, const vec3& v2, const vec3& v3,int id)
	{
		_id = id;
		p1 = Vector3(v1.X(),v1.Y(),v1.Z());
		p2 = Vector3(v2.X(),v2.Y(),v2.Z());
		p3 = Vector3(v3.X(),v3.Y(),v3.Z());
		vec3 center = (v1 + v2 + v3)/3;
		_center = Vector3(center.X(),center.Y(),center.Z());
		
	}
	Triangle(const Vector3& v1, const Vector3& v2, const Vector3& v3):p1(v1),p2(v2),p3(v3)
	{
	}

	virtual bool getIntersection(
		const Ray& ray, 
		IntersectionInfo* intersection)  const
	{
		//not implemented
		return true;
	}

	//! Return an object normal based on an intersection
	virtual Vector3 getNormal(const IntersectionInfo& I) const
	{
		return Vector3((p3-p2) ^(p1-p2));
	}

	//! Return a bounding box for this object
	virtual BBox getBBox() const
	{ 
	/*	BBox _bbox(p1);
		_bbox.expandToInclude(p2);
		_bbox.expandToInclude(p3);*/
		float minX,maxX,minY,maxY,minZ,maxZ;
		if (p1.x < p2.x)
		{
			minX = p1.x;
			maxX = p2.x;
		}
		else
		{
			minX = p2.x;
			maxX = p1.x;
		}
		if (p1.y < p2.y)
		{
			minY = p1.y;
			maxY = p2.y;
		}
		else
		{
			minY = p2.y;
			maxY = p1.y;
		}
		if (p1.z < p2.z)
		{
			minZ = p1.z;
			maxZ = p2.z;
		}
		else
		{
			minZ = p2.z;
			maxZ = p1.z;
		}
		if (p3.x < minX)
			minX = p3.x;
		if(p3.y < minY)
			minY = p3.y;
		if(p3.z < minZ)
			minZ = p3.z;
		if(p3.x > maxX)
			maxX = p3.x;
		if(p3.y > maxY)
			maxY = p3.y;
		if(p3.z >maxZ)
			maxZ = p3.z;
		return BBox(Vector3(minX,minY,minZ),Vector3(maxX,maxY,maxZ));
	}

	//! Return the centroid for this object. (Used in BVH Sorting)
	virtual Vector3 getCentroid() const
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
	Vector3 _center;
public:
	Box(){}
	Vector3 min,max;
	Box(const vec3& m, const vec3& M, int id):_id(id)
	{
		min.x = m.X();
		min.y = m.Y();
		min.z = m.Z();
		max.x = M.X();
		max.y = M.Y();
		max.z = M.Z();
		vec3 center = (m+M)/2;
		_center = Vector3(center.X(),center.Y(),center.Z());
		
	}
	
	virtual bool getIntersection(
		const Ray& ray, 
		IntersectionInfo* intersection)  const
	{
		//not implemented
		return true;
	}

	//! Return an object normal based on an intersection
	virtual Vector3 getNormal(const IntersectionInfo& I) const
	{
		return Vector3(0,0,0);
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
	virtual Vector3 getCentroid() const
	{
		return _center;
	}

	int getId() const
	{
		return _id;
	}
};
#endif