
struct cullingContext
{
	__host__ __device__ cullingContext()
	{
		triId = uint32(-1);
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
struct is_frustum
{
	__host__ __device__ bool operator()(const TriFrustum& c)
	{
		return c.id != uint32(-1);
	}
};