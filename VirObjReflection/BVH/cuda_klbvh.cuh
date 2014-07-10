#pragma once
struct cullingContext
{
	__host__ __device__ cullingContext()
	{
		triId = uint32(-1);
	}
	__host__ __device__ cullingContext(uint32 f, uint32 t)
	{
		frustumId = f;
		triId = t;
	}
	uint32 frustumId;
	uint32 triId;
};
struct is_Triangle
{
	__host__ __device__ is_Triangle(uint32 id):_id(id)
	{}
	__host__ __device__ bool operator()(const cullingContext& c)
	{
		return c.triId == _id;
	}
	uint32 _id;
};
struct is_Frustum
{
	__host__ __device__ is_Frustum(uint32 id):_id(id){}
	__host__ __device__ bool operator()(const cullingContext& c)
	{
		return c.frustumId == _id;
	}
	uint32 _id;
};
struct is_valid
{
	static const  uint32 invalid = uint32(-1);
	__host__ __device__ bool operator()(const cullingContext& c)
	{
		return c.triId != invalid;
	}
};
struct is_inValid
{
	static const  uint32 invalid = uint32(-1);
	__host__ __device__ bool operator()(const cullingContext& c)
	{
		return c.triId == invalid;
	}
};
struct is_frustum
{
	__host__ __device__ bool operator()(const TriFrustum& c)
	{
		return c.id == uint32(-1);
	}
};
struct is_negtive
{
	__host__ __device__ bool operator()(const unsigned& c)
	{
		return c == uint32(-1);
	}
};
__device__  __host__ inline nih::Vector3f MatrixXVector3f(const float* mat, nih::Vector3f& vec)
{
		float* d= &vec[0];
		float tmp[4];
		tmp[0] = mat[0]*d[0] + mat[4]*d[1] + mat[8]*d[2] + mat[12];
		tmp[1] = mat[1]*d[0] + mat[5]*d[1] + mat[9]*d[2] + mat[13];
		tmp[2] = mat[2]*d[0] + mat[6]*d[1] + mat[10]*d[2] + mat[14];
		tmp[3] = mat[3]*d[0] + mat[7]*d[1] + mat[11]*d[2] + mat[15];

		Vector3f ret( tmp );
		ret /= tmp[3];
		return ret;
	}