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

//opengl standard frustum
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
	virtual inline CullingResult ContainsTriangle(Triangle* tri)
	{
		vec4 v1(tri->p1.X(),tri->p1.Y(),tri->p1.Z(),1);
		vec4 v2(tri->p2.X(),tri->p2.Y(),tri->p2.Z(),1);
		vec4 v3(tri->p3.X(),tri->p3.Y(),tri->p3.Z(),1);
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
	virtual inline CullingResult ContainsBBox(const BBox& refBox) const
	{

		/* create the bounding box of the object */
		vec4 BoundingBox[8];
		BoundingBox[0] = _MVP * vec4(refBox.min.X(),refBox.min.Y(),refBox.min.Z(),1);
		BoundingBox[1] = _MVP * vec4(refBox.min.X(),refBox.min.Y(),refBox.max.Z(),1);
		BoundingBox[2] = _MVP * vec4(refBox.min.X(),refBox.max.Y(),refBox.min.Z(),1);
		BoundingBox[3] = _MVP * vec4(refBox.min.X(),refBox.max.Y(),refBox.max.Z(),1);
		BoundingBox[4] = _MVP * vec4(refBox.max.X(),refBox.min.Y(),refBox.min.Z(),1);
		BoundingBox[5] = _MVP * vec4(refBox.max.X(),refBox.min.Y(),refBox.max.Z(),1);
		BoundingBox[6] = _MVP * vec4(refBox.max.X(),refBox.max.Y(),refBox.min.Z(),1);
		BoundingBox[7] = _MVP * vec4(refBox.max.X(),refBox.max.Y(),refBox.max.Z(),1);

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
	inline CullingResult ContainsBox(const Box& refBox) const
	{
		/* create the bounding box of the object */
		vec4 BoundingBox[8];
		BoundingBox[0] = _MVP * vec4(refBox.min.X(),refBox.min.Y(),refBox.min.Z(),1);
		BoundingBox[1] = _MVP * vec4(refBox.min.X(),refBox.min.Y(),refBox.max.Z(),1);
		BoundingBox[2] = _MVP * vec4(refBox.min.X(),refBox.max.Y(),refBox.min.Z(),1);
		BoundingBox[3] = _MVP * vec4(refBox.min.X(),refBox.max.Y(),refBox.max.Z(),1);
		BoundingBox[4] = _MVP * vec4(refBox.max.X(),refBox.min.Y(),refBox.min.Z(),1);
		BoundingBox[5] = _MVP * vec4(refBox.max.X(),refBox.min.Y(),refBox.max.Z(),1);
		BoundingBox[6] = _MVP * vec4(refBox.max.X(),refBox.max.Y(),refBox.min.Z(),1);
		BoundingBox[7] = _MVP * vec4(refBox.max.X(),refBox.max.Y(),refBox.max.Z(),1);

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
	virtual void Draw();
	virtual void CreateVAO();
private:
	IGLUMatrix4x4 _PM;
	IGLUMatrix4x4 _VM;
	IGLUMatrix4x4 _MVP;
	IGLUVertexArray::Ptr _vao;
	vec3 _eye,_at,_up;
	float _fovy,_aspectRatio,_near,_far;
};

//frustum with 5 planes
typedef struct plane
{
	plane(){}
	//已知三点求平面方程, 右手坐标系
	plane(const vec3& p1, const vec3& p2, const vec3& p3)
	{
		vec3 norm = (p3-p2).Cross(p1-p2);
		norm.Normalize();
		a = norm.X();
		b = norm.Y();
		c = norm.Z();
		d = -p1.Dot(norm);

	}

	//点到平面的距离
	float distance(const vec3& p)
	{
		return p.Dot(vec3(a,b,c))+d;
	}

	//求经过点p0和p1的直线和平面交点
	bool intersect(const vec3& p0, const vec3& p1, vec3& intersec)
	{
		vec3 dir = p1-p0;
		dir.Normalize();
		vec3 fnormal(a,b,c);

		float v = fnormal.Dot(dir);
		if (fabs(v)<IGLU_ZERO)
			return false;
		
		float t = (p0.Dot(fnormal)+d)/-v;
		intersec = p0 + dir*t;
		return true;
	}
	float a;
	float b;
	float c;
	float d;
} plane_t;
//

//Pyramidal Frustum
typedef struct pyrfrustum
{
	plane_t planes[6];
} pyrfrustum_t;

class TriFrustum : public Frustum
{
public:
	TriFrustum():_vao(NULL){}
	//视点，三角形，远裁剪
	TriFrustum(vec3& eye, Triangle& tri, float farD):_eye(eye),_tri(tri),_far(farD)
	{
		//求5个平面方程
		//视锥平面法线指向视锥外
		plane_t pTri(tri.p1,tri.p2,tri.p3);
		memcpy(_points, eye.GetDataPtr(),sizeof(float)*3);

		float d  = pTri.distance(eye);
		//视点不能位于三角形平面内
		if (fabs(d) < IGLU_ZERO)
			fprintf(stderr,"create trifrustum error, eye is in the plane of triangle");

		if (d > 0)
		{
			memcpy(_points+3, tri.p1.GetDataPtr(),sizeof(float)*3);
			memcpy(_points+6, tri.p2.GetDataPtr(),sizeof(float)*3);
			memcpy(_points+9, tri.p3.GetDataPtr(),sizeof(float)*3);
			_planes[0] = plane_t(eye,tri.p1,tri.p2);
			_planes[1] = plane_t(eye,tri.p2,tri.p3);
			_planes[2] = plane_t(eye,tri.p3,tri.p1);
			_planes[3] = pTri;
			_planes[4] = plane_t(tri.p1,tri.p3,tri.p2);
			vec3 c = tri.getCentroid();
			float cosT = d/(eye-c).Length();
			_planes[4].d -= _far/cosT;

			vec3 p4,p5,p6;
			assert(_planes[4].intersect(eye,tri.p1,p4));
			memcpy(_points+12, p4.GetDataPtr(),sizeof(float)*3);
			assert(_planes[4].intersect(eye,tri.p2,p5));
			memcpy(_points+15, p5.GetDataPtr(),sizeof(float)*3);
			assert(_planes[4].intersect(eye,tri.p3,p6));
			memcpy(_points+18, p6.GetDataPtr(),sizeof(float)*3);
		}
		else
		{
			memcpy(_points+3, tri.p1.GetDataPtr(),sizeof(float)*3);
			memcpy(_points+6, tri.p3.GetDataPtr(),sizeof(float)*3);
			memcpy(_points+9, tri.p2.GetDataPtr(),sizeof(float)*3);
			_planes[0] = plane_t(eye,tri.p2,tri.p1);
			_planes[1] = plane_t(eye,tri.p3,tri.p2);
			_planes[2] = plane_t(eye,tri.p1,tri.p3);
			_planes[3] =  plane_t(tri.p1,tri.p3,tri.p2);
			_planes[4] = pTri;
			vec3 c = tri.getCentroid();
			float cosT = (-d)/(eye-c).Length();
			_planes[4].d -= _far/cosT;

			vec3 p4,p5,p6;
			assert(_planes[4].intersect(eye,tri.p1,p4));
			memcpy(_points+12, p4.GetDataPtr(),sizeof(float)*3);
			assert(_planes[4].intersect(eye,tri.p3,p5));
			memcpy(_points+15, p5.GetDataPtr(),sizeof(float)*3);
			assert(_planes[4].intersect(eye,tri.p2,p6));
			memcpy(_points+18, p6.GetDataPtr(),sizeof(float)*3);
		}
		_vao = NULL;
	}
	~TriFrustum()
	{
		safe_delete(_vao);
	}
	virtual void CreateVAO();
	virtual void Draw();
	virtual inline CullingResult ContainsTriangle(Triangle* tri)
	{
		
		bool intersect = false;
		int out = 0;
		for(int i=0; i<5; i++)
		{
			if (_planes[i].distance(tri->p1)>0)
				out++;
			if (_planes[i].distance(tri->p2)>0)
				out++;
			if (_planes[i].distance(tri->p3)>0)
				out++;
			if (out == 3)
				return Out;
			else if (out >0)
				intersect = true;

			out = 0;
				
		}
		return intersect? Intersect : In;
	}
	virtual inline CullingResult ContainsBBox(const BBox& refBox) const
	{
		bool intersec = false;
		unsigned tableXX[2] = {3,0};
		unsigned tableYY[2] = {4,1};
		unsigned tableZZ[2] = {5,2};
		unsigned tableX[2] = {0,3};
		unsigned tableY[2] = {1,4};
		unsigned tableZ[2] = {2,5};
		float p[6];
		memcpy(p,refBox.min.GetConstDataPtr(),sizeof(float)*3);
		memcpy(p+3,refBox.max.GetConstDataPtr(),sizeof(float)*3);
		for(int i=0; i<5; i++)
		{
			//plane_t plane= f.planes[i];
			unsigned sa = _planes[i].a > 0; 
			unsigned sb = _planes[i].b >0; 
			unsigned sc = _planes[i].c > 0;
			if (p[tableX[sa]]*_planes[i].a + p[tableY[sb]]*_planes[i].b + p[tableZ[sc]]*_planes[i].c+_planes[i].d >=0)
				return Out;
			if (p[tableXX[sa]]*_planes[i].a + p[tableYY[sb]]*_planes[i].b + p[tableZZ[sc]]*_planes[i].c+_planes[i].d >=0)
				intersec = true;

		}
		return intersec ? Intersect : In; 
		
	}
private:
	vec3 _eye;
	Triangle _tri;
	float _far;
	plane_t _planes[5];
	float _points[21];
	IGLUVertexArray::Ptr _vao;
};

#endif