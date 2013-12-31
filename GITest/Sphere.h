#ifndef Sphere_h_
#define Sphere_h_

#include <cmath>
#include "Object.h"

//! For the purposes of demonstrating the BVH, a simple sphere
struct Sphere : public Object {
 vec3 center; // Center of the sphere
 vec3 p1,p2,p3;//some tests
 float r, r2; // Radius, Radius^2
 Sphere(const vec3& p1,const vec3& p2, const vec3& p3)
	 : p1(p1),p2(p2),p3(p3){ }
 Sphere(const vec3& center, float radius)
 : center(center), r(radius), r2(radius*radius) { }

 bool getIntersection(const Ray& ray, IntersectionInfo* I) const {
	 vec3 c(center);
  vec3 s = c - ray.o;
  float sd = s.Dot(ray.d);
  float ss = s.Dot(s);
  
  // Compute discriminant
  float disc = sd*sd - ss + r2;

  // Complex values: No intersection
  if( disc < 0.f ) return false; 

  // Assume we are not in a sphere... The first hit is the lesser valued
  I->object = this;
  I->t = sd - sqrt(disc);
  return true;
 }
 
 vec3 getNormal(const IntersectionInfo& I) const {
	 vec3 c(center);
  return Normalize(I.hit - c);
 }

 BBox getBBox() const { 
	vec3 c(center);
  return BBox(c-vec3(r,r,r), c+vec3(r,r,r)); 
 }

 vec3 getCentroid() const {
  //return center;
	 return (p1 + p2 + p3)/3;
 }

};

#endif
