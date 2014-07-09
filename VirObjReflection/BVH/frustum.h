#pragma once
#include "types.h"
#include "vector.h"
#include "bbox.h"
namespace nih
{

	//Frustum plane
	typedef struct plane
	{
		NIH_HOST_DEVICE plane(){}
		NIH_HOST_DEVICE plane(const Vector3f& p1, const Vector3f& p2, const Vector3f& p3)
		{
			Vector3f norm = cross((p3-p2),(p1-p2));
			norm = normalize(norm);
			a = norm[0];
			b = norm[1];
			c = norm[2];
			d = -dot(p1,norm);
		}
		//点到平面的距离
		NIH_HOST_DEVICE float distance(const Vector3f& p)
		{
			return dot(p,Vector3f(a,b,c))+d;
		}

		//求经过点p0和p1的直线和平面交点
		NIH_HOST_DEVICE bool intersect(const Vector3f& p0, const Vector3f& p1, Vector3f& intersec)
		{
			Vector3f dir = p1-p0;
			dir = normalize(dir);
			Vector3f fnormal(a,b,c);

			float v = dot(fnormal,dir);
			if (abs(v)<1e-4)
				return false;

			float t = (dot(p0,fnormal)+d)/-v;
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

	//Tri Frustum
	typedef struct TriFrustum
	{
		TriFrustum()
		{
			id = uint32(-1);
		}
		uint32 id;
		plane_t planes[5];
	} TriFrustum;


	//Pyramidal Frustum Corners
	typedef struct pyrcorners
	{
		Vector3f points[8];
	} pyrcorners_t;

	FORCE_INLINE NIH_HOST_DEVICE void NormalizePlane(plane_t & plane)
	{
		float mag;
		mag = sqrt(plane.a * plane.a + plane.b * plane.b + plane.c * plane.c);
		plane.a = plane.a / mag;
		plane.b = plane.b / mag;
		plane.c = plane.c / mag;
		plane.d = plane.d / mag;
	}
	FORCE_INLINE NIH_HOST_DEVICE float planeDistance( Vector3f & v, plane_t& p )
	{
		return (v[0]*p.a + v[1]*p.b + v[2]*p.c + p.d );
	}
	FORCE_INLINE NIH_HOST_DEVICE int Intersect( TriFrustum& f, Vector3f& p1, Vector3f& p2, Vector3f p3 )
	{
		int TotalIn = 0;
		for(int i=0; i<5;i++)
		{
			int in = 3;
			int iPtIn = 1;
			if( planeDistance(p1,f.planes[i]) > 0)
			{
				in--;
				iPtIn = 0;
			}
			if( planeDistance(p2,f.planes[i]) > 0)
			{
				in--;
				iPtIn = 0;
			}
			if( planeDistance(p3,f.planes[i]) > 0)
			{
				in--;
				iPtIn = 0;
			}
			if (in ==0)
				return 0;

			TotalIn +=iPtIn;
		}
		if(TotalIn == 3)
			return 1;

		// we must be partly in then otherwise
		return 2;
	}
	FORCE_INLINE NIH_HOST_DEVICE int Intersect( TriFrustum& f, float*ptr )
	{

		Vector3f box[8];
		box[0][0] = ptr[0]; box[0][1] = ptr[1]; box[0][2] = ptr[2];
		box[1][0] = ptr[3]; box[1][1] = ptr[1]; box[1][2] = ptr[2];
		box[2][0] = ptr[0]; box[2][1] = ptr[4]; box[2][2] = ptr[2];
		box[3][0] = ptr[3]; box[3][1] = ptr[4]; box[3][2] = ptr[2];
		box[4][0] = ptr[0]; box[4][1] = ptr[1]; box[4][2] = ptr[5];
		box[5][0] = ptr[3]; box[5][1] = ptr[1]; box[5][2] = ptr[5];
		box[6][0] = ptr[0]; box[6][1] = ptr[4]; box[6][2] = ptr[5];
		box[7][0] = ptr[3]; box[7][1] = ptr[4]; box[7][2] = ptr[5];

		int iTotalIn = 0;

		// test all 8 corners against the 6 sides 
		// if all points are behind 1 specific plane, we are out
		// if we are in with all points, then we are fully in
		for(int p = 0; p < 5; ++p) {

			int iInCount = 8;
			int iPtIn = 1;

			for(int i = 0; i < 8; ++i) {

				// test this point against the planes
				if(planeDistance( box[i], f.planes[p] ) > 0 ) {
					iPtIn = 0;
					--iInCount;
				}
			}

			// were all the points outside of plane p?
			if(iInCount == 0)
				return 0;

			// check if they were all on the right side of the plane
			iTotalIn += iPtIn;
		}

		// so if iTotalIn is 6, then all are inside the view
		if(iTotalIn == 5)
			return 1;

		// we must be partly in then otherwise
		return 2;
	}

	struct Matrix4x4
	{
		// The elements of the 4x4 matrix are stored in
		// column-major order (see "OpenGL Programming Guide",
		// 3rd edition, pp 106, glLoadMatrix).
		float _11, _21, _31, _41;
		float _12, _22, _32, _42;
		float _13, _23, _33, _43;
		float _14, _24, _34, _44;
	};
	FORCE_INLINE NIH_HOST_DEVICE void ExtractPlanesGL(
		plane_t * p_planes,
		const Matrix4x4 & comboMatrix,
		bool normalize)
	{
		// Left clipping plane
		p_planes[0].a = comboMatrix._41 + comboMatrix._11;
		p_planes[0].b = comboMatrix._42 + comboMatrix._12;
		p_planes[0].c = comboMatrix._43 + comboMatrix._13;
		p_planes[0].d = comboMatrix._44 + comboMatrix._14;
		// Right clipping plane
		p_planes[1].a = comboMatrix._41 - comboMatrix._11;
		p_planes[1].b = comboMatrix._42 - comboMatrix._12;
		p_planes[1].c = comboMatrix._43 - comboMatrix._13;
		p_planes[1].d = comboMatrix._44 - comboMatrix._14;
		// Top clipping plane
		p_planes[2].a = comboMatrix._41 - comboMatrix._21;
		p_planes[2].b = comboMatrix._42 - comboMatrix._22;
		p_planes[2].c = comboMatrix._43 - comboMatrix._23;
		p_planes[2].d = comboMatrix._44 - comboMatrix._24;
		// Bottom clipping plane
		p_planes[3].a = comboMatrix._41 + comboMatrix._21;
		p_planes[3].b = comboMatrix._42 + comboMatrix._22;
		p_planes[3].c = comboMatrix._43 + comboMatrix._23;
		p_planes[3].d = comboMatrix._44 + comboMatrix._24;
		// Near clipping plane
		p_planes[4].a = comboMatrix._41 + comboMatrix._31;
		p_planes[4].b = comboMatrix._42 + comboMatrix._32;
		p_planes[4].c = comboMatrix._43 + comboMatrix._33;
		p_planes[4].d = comboMatrix._44 + comboMatrix._34;
		// Far clipping plane
		p_planes[5].a = comboMatrix._41 - comboMatrix._31;
		p_planes[5].b = comboMatrix._42 - comboMatrix._32;
		p_planes[5].c = comboMatrix._43 - comboMatrix._33;
		p_planes[5].d = comboMatrix._44 - comboMatrix._34;
		// Normalize the plane equations, if requested
		if (normalize == true)
		{
			NormalizePlane(p_planes[0]);
			NormalizePlane(p_planes[1]);
			NormalizePlane(p_planes[2]);
			NormalizePlane(p_planes[3]);
			NormalizePlane(p_planes[4]);
			NormalizePlane(p_planes[5]);
		}
	}
}


