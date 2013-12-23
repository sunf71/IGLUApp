#ifndef FRUSTUM_H
#define FRUSTUM_H
#pragma warning( disable: 4996 )

#include "iglu.h"
#include "BBox.h"
#include "Triangle.h"
/*
Frustum class
A view frustum is defined by six planes
*/
using namespace iglu;

enum CullingResult
{
	In = 1,
	Intersect = 0,
	Out=-1
};

struct AaBox
{
	float minX,maxX,minY,maxY,minZ,maxZ;
};

class Frustum
{
public :
	Frustum()
	{
		
	}
	~Frustum()
	{
		if (_vao != NULL)
			delete _vao;
	}
	Frustum(const vec3 &eye, const vec3 &at, const vec3& up,  
		float fovy, float nearZ, float farZ, float aspectRatio):_eye(eye),_at(at),_up(up),_fovy(fovy),_near(nearZ),_far(farZ),_aspectRatio(aspectRatio)
	{
		_PM = IGLUMatrix4x4::Perspective(fovy,aspectRatio,nearZ,farZ);	
		_VM = IGLUMatrix4x4::LookAt(eye,at,up);
		_MVP = _PM * _VM ;
		_vao = NULL;
	}
	Frustum( const IGLUMatrix4x4 & pm, IGLUMatrix4x4 & vm):_PM(pm),_VM(vm)
	{
		_MVP = _PM * _VM;
		_vao = NULL;
	}
	inline CullingResult ContainsTriangle(Triangle* tri)
	{
		vec4 v1(tri->p1.x,tri->p1.y,tri->p1.z,1);
		vec4 v2(tri->p2.x,tri->p2.y,tri->p2.z,1);
		vec4 v3(tri->p3.x,tri->p3.y,tri->p3.z,1);
		v1 = _MVP * v1;
		v2 = _MVP * v2;
		v3 = _MVP * v3;
		int outOfBound[6] = {0};
		vec4 BoundingBox[3];
		BoundingBox[0] = v1;
		BoundingBox[1] = v2;
		BoundingBox[2] = v3;
		for (int i=0; i<3; i++)
		{
			if ( BoundingBox[i].X() >  BoundingBox[i].W() ) outOfBound[0]++;
			if ( BoundingBox[i].X() < -BoundingBox[i].W() ) outOfBound[1]++;
			if ( BoundingBox[i].Y() >  BoundingBox[i].W() ) outOfBound[2]++;
			if ( BoundingBox[i].Y() < -BoundingBox[i].W() ) outOfBound[3]++;
			if ( BoundingBox[i].Z() >  BoundingBox[i].W() ) outOfBound[4]++;
			if ( BoundingBox[i].Z() < -BoundingBox[i].W() ) outOfBound[5]++;
		}

		bool inFrustum = true;

		for (int i=0; i<6; i++)
		{
			if ( outOfBound[i] >0)
				inFrustum = false;

			if ( outOfBound[i] == 3 ) 
				return Out;
		}

		return inFrustum ? In : Intersect;
	}
	inline CullingResult ContainsBBox(const BBox& refBox) const
	{
	
		/* create the bounding box of the object */
		vec4 BoundingBox[8];
		BoundingBox[0] = _MVP * vec4(refBox.min.x,refBox.min.y,refBox.min.z,1);
		BoundingBox[1] = _MVP * vec4(refBox.min.x,refBox.min.y,refBox.max.z,1);
		BoundingBox[2] = _MVP * vec4(refBox.min.x,refBox.max.y,refBox.min.z,1);
		BoundingBox[3] = _MVP * vec4(refBox.min.x,refBox.max.y,refBox.max.z,1);
		BoundingBox[4] = _MVP * vec4(refBox.max.x,refBox.min.y,refBox.min.z,1);
		BoundingBox[5] = _MVP * vec4(refBox.max.x,refBox.min.y,refBox.max.z,1);
		BoundingBox[6] = _MVP * vec4(refBox.max.x,refBox.max.y,refBox.min.z,1);
		BoundingBox[7] = _MVP * vec4(refBox.max.x,refBox.max.y,refBox.max.z,1);

		/* check how the bounding box resides regarding to the view frustum */   
		int outOfBound[6] = {0};

		for (int i=0; i<8; i++)
		{
			if ( BoundingBox[i].X() >  BoundingBox[i].W() ) outOfBound[0]++;
			if ( BoundingBox[i].X() < -BoundingBox[i].W() ) outOfBound[1]++;
			if ( BoundingBox[i].Y() >  BoundingBox[i].W() ) outOfBound[2]++;
			if ( BoundingBox[i].Y() < -BoundingBox[i].W() ) outOfBound[3]++;
			if ( BoundingBox[i].Z() >  BoundingBox[i].W() ) outOfBound[4]++;
			if ( BoundingBox[i].Z() < -BoundingBox[i].W() ) outOfBound[5]++;
		}

		bool inFrustum = true;

		for (int i=0; i<6; i++)
		{
			if ( outOfBound[i] >0)
				inFrustum = false;

			if ( outOfBound[i] == 8 ) 
				return Out;
		}

		return inFrustum ? In : Intersect;
	}
	inline CullingResult ContainsAaBox(const AaBox& refBox) const
	{
		/* create the bounding box of the object */
		vec4 BoundingBox[8];
		BoundingBox[0] = _MVP * vec4(refBox.minX,refBox.minY,refBox.minZ,1);
		BoundingBox[1] = _MVP * vec4(refBox.minX,refBox.minY,refBox.maxZ,1);
		BoundingBox[2] = _MVP * vec4(refBox.minX,refBox.maxY,refBox.minZ,1);
		BoundingBox[3] = _MVP * vec4(refBox.minX,refBox.maxY,refBox.maxZ,1);
		BoundingBox[4] = _MVP * vec4(refBox.maxX,refBox.minY,refBox.minZ,1);
		BoundingBox[5] = _MVP * vec4(refBox.maxX,refBox.minY,refBox.maxZ,1);
		BoundingBox[6] = _MVP * vec4(refBox.maxX,refBox.maxY,refBox.minZ,1);
		BoundingBox[7] = _MVP * vec4(refBox.maxX,refBox.maxY,refBox.maxZ,1);

		/* check how the bounding box resides regarding to the view frustum */   
		int outOfBound[6] = {0};

		for (int i=0; i<8; i++)
		{
			if ( BoundingBox[i].X() >  BoundingBox[i].W() ) outOfBound[0]++;
			if ( BoundingBox[i].X() < -BoundingBox[i].W() ) outOfBound[1]++;
			if ( BoundingBox[i].Y() >  BoundingBox[i].W() ) outOfBound[2]++;
			if ( BoundingBox[i].Y() < -BoundingBox[i].W() ) outOfBound[3]++;
			if ( BoundingBox[i].Z() >  BoundingBox[i].W() ) outOfBound[4]++;
			if ( BoundingBox[i].Z() < -BoundingBox[i].W() ) outOfBound[5]++;
		}

		bool inFrustum = true;

		for (int i=0; i<6; i++)
		{
			if ( outOfBound[i] >0)
				inFrustum = false;

			if ( outOfBound[i] == 8 ) 
				return Out;
		}

		return inFrustum ? In : Intersect;
	}
	void Draw();
	void CreateVAO();
private:
	IGLUMatrix4x4 _PM;
	IGLUMatrix4x4 _VM;
	IGLUMatrix4x4 _MVP;
	IGLUVertexArray::Ptr _vao;
	vec3 _eye,_at,_up;
	float _fovy,_aspectRatio,_near,_far;
};

#endif