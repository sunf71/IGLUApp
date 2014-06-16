#ifndef IGLUMATHELP_H
#define IGLUMATHHELP_H
#include "iglu.h"
using namespace iglu;
class IGLUMathHelper
{
public:
	//求直线点p，方向dir与平面（A，B，C三点在平面上）的交点
	//grab from http://blog.csdn.net/abcjennifer/article/details/6688080
	static  bool LineIntersecPlane(vec3& p, vec3& dir, vec3& A, vec3& B, vec3& C, vec3& intersec)
	{
		vec3 normal = ((C-B).Cross(A-B));
		normal.Normalize();

		float vp1, vp2, vp3, n1, n2, n3, v1, v2, v3, m1, m2, m3, t,vpt;
		vp1 = normal[0];
		vp2 = normal[1];
		vp3 = normal[2];
		n1 = A[0];
		n2 = A[1];
		n3 = A[2];
		v1 = dir[0];
		v2 = dir[1];
		v3 = dir[2];
		m1 = p[0];
		m2 = p[1];
		m3 = p[2];
		vpt = dir.Dot(normal);
		//首先判断直线是否与平面平行
		if (fabs(vpt) < IGLU_ZERO)
		{
			return false;
		}
		else
		{
			t = ((n1 - m1) * vp1 + (n2 - m2) * vp2 + (n3 - m3) * vp3) / vpt;
			intersec[0] = m1 + v1 * t;
			intersec[1] = m2 + v2 * t;
			intersec[2] = m3 + v3 * t;
		}
		return true;
	}
	//求直线点p，方向dir与平面（A,normal）的交点
	static  bool LineIntersecPlane(vec3& p, vec3& dir, vec3& A, vec3& normal, vec3& intersec)
	{
		
		float vp1, vp2, vp3, n1, n2, n3, v1, v2, v3, m1, m2, m3, t,vpt;
		vp1 = normal[0];
		vp2 = normal[1];
		vp3 = normal[2];
		n1 = A[0];
		n2 = A[1];
		n3 = A[2];
		v1 = dir[0];
		v2 = dir[1];
		v3 = dir[2];
		m1 = p[0];
		m2 = p[1];
		m3 = p[2];
		vpt = dir.Dot(normal);
		//首先判断直线是否与平面平行
		if (fabs(vpt) < IGLU_ZERO)
		{
			return false;
		}
		else
		{
			t = ((n1 - m1) * vp1 + (n2 - m2) * vp2 + (n3 - m3) * vp3) / vpt;
			intersec[0] = m1 + v1 * t;
			intersec[1] = m2 + v2 * t;
			intersec[2] = m3 + v3 * t;
		}
		return true;
	}
	//Determine whether point P in triangle ABC
	//grab from somewhere searched by google
	bool PointInTriangle(vec3& A, vec3& B, vec3& C, vec3& P)
	{
		vec3 v0 = C - A ;
		vec3 v1 = B - A ;
		vec3 v2 = P - A ;

		float dot00 = v0.Dot(v0) ;
		float dot01 = v0.Dot(v1) ;
		float dot02 = v0.Dot(v2) ;
		float dot11 = v1.Dot(v1) ;
		float dot12 = v1.Dot(v2) ;

		float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01) ;

		float u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;
		if (u < 0 || u > 1) // if u out of range, return directly
		{
			return false ;
		}

		float v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
		if (v < 0 || v > 1) // if v out of range, return directly
		{
			return false ;
		}

		return u + v <= 1 ;
	}

	
};



#endif