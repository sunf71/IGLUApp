#ifndef Ray_h
#define Ray_h

#include "iglu.h"
using namespace iglu;
struct Ray {
 vec3 o; // Ray Origin
 vec3 d; // Ray Direction
 vec3 inv_d; // Inverse of each Ray Direction component

 Ray(const vec3& o, const vec3& d)
 : o(o), d(d), inv_d(d) { }
};

#endif
