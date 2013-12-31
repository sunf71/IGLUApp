#ifndef BBox_h
#define BBox_h

#include "Ray.h"
#include "iglu.h"
#include <stdint.h>
using namespace iglu;
struct BBox {
	vec3 min, max, extent;
	BBox() { }
	BBox(const vec3& min, const vec3& max);
	BBox(const vec3& p);

	bool intersect(const Ray& ray, float *tnear, float *tfar) const;
 void expandToInclude(const vec3& p);
 void expandToInclude(const BBox& b);
 uint32_t maxDimension() const;
 float surfaceArea() const;
};

#endif
