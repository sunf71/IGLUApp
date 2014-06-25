#include "CudaFunctions.h"
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include "BVH\cuda_klbvh.h"
#include "BVH\cuda_klbvh.cuh"
#include <iostream>
texture<uint32> bvhTex;
__global__ void GenerateVirFrustumKernel(Vector3f* eye,Vector3f* p123, TriFrustum* frustums,float farD, int count);
__global__ void FrustumCullingKernel(TriFrustum* frustumP, int frustum_num, uint32 priSize,cullingContext* list);
namespace cuda
{
	
	size_t GridSize(size_t jobSize,size_t blockSize = 128)
	{

		int numSMs;
		cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		size_t max_blocks = 65535;
		return numSMs*nih::min( max_blocks, numSMs*(jobSize + (blockSize*numSMs)-1) / (blockSize*numSMs) );
	}

	//@vertices �Ƕ������飬ÿ��������4����������ʾ
	//@indices �������ζ�����������飬ÿ����������3������
	//@matrix ��4��4ģ�;���
	//@points ������������������ζ���
	//@centers ������������������ε�����
	//@boxes ������������������εİ�Χ��
	//@size �����θ���
	__global__ void LoadOBJKernel(float* vertices,
		uint32* indices,
		float* matrix,
		nih::Vector3f* points,
		nih::Vector3f* centers,
		nih::Bbox3f* boxes,
		uint32 size)
	{
		uint32 step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < size; 
			i += step) 
		{
			uint32 tOffset = 3*i;


			for( int j=0; j<3; j++)
			{
				for( int k=0; k<3; k++)
				{
					points[tOffset+j][k] = vertices[indices[tOffset+j]*3+k];
					centers[i][k] += points[tOffset+j][k];
				}
				boxes[i].insert(points[tOffset+j]);				
			}			
			centers[i] /= 3.f;
		}
	}


	void LoadOBJReader(iglu::IGLUOBJReader::Ptr obj, 
		iglu::IGLUMatrix4x4::Ptr matrix,
		nih::Vector3f* d_points,
		nih::Vector3f* d_centers,
		nih::Bbox3f* d_boxes)
	{
		size_t size = obj->GetTriangleCount();
		size_t vertSize = obj->GetVaoVerts().size();
		float* vertices = (float*)(&obj->GetVaoVerts()[0]);
		uint32* indices = obj->GetElementArrayData();
		for(int i=0; i<size*3; i+=3)
			std::cout<<indices[i]<<","<<indices[i+1]<<","<<indices[i+2]<<std::endl;
		float* d_vertices;
		cudaMalloc((void**)&d_vertices,sizeof(float)*12*size);
		cudaMemcpy(d_vertices,vertices,sizeof(float)*12*size,cudaMemcpyHostToDevice);

		uint32* d_indices;
		cudaMalloc((void**)&d_indices,sizeof(uint32)*3*size);
		cudaMemcpy(d_indices,indices,sizeof(uint32)*3*size,cudaMemcpyHostToDevice);

		/*thrust::host_vector<nih::Vector3f> h_points(size*3);
		thrust::host_vector<nih::Vector3f> h_centers(size);
		thrust::host_vector<nih::Bbox3f> h_boxes(size);
		nih::Bbox3f gbox;
		for(int i=0; i<size; i++)
		{
		uint32 tOffset = i*3;
		for( int j=0; j<3; j++)
		{
		for( int k=0; k<3; k++)
		{
		h_points[tOffset+j][k] = vertices[indices[tOffset+j]*3+k];
		h_centers[i][k] += h_points[tOffset+j][k];
		}
		h_boxes[i].insert(h_points[tOffset+j]);
		}			
		}
		for(int i=0; i<h_points.size(); i++)
		std::cout<<h_points[i][0]<<","<<h_points[i][1]<<","<<h_points[i][2]<<std::endl;
		*/
		size_t n_blocks = GridSize(size);
		LoadOBJKernel<<<n_blocks,128>>>(d_vertices,d_indices,&matrix[0][0],d_points,d_centers,d_boxes,size);
		cudaFree(d_indices);
		cudaFree(d_vertices);
	}

	//
	size_t BuildBvh(iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize)
	{
		using namespace nih;
		uint32 size = 0;
		uint32 *offsets = new uint32[objSize+1];

		for(int i=0; i<objSize; i++)
		{
			offsets[i] = size;
			size += objs[i]->GetTriangleCount();

		}
		thrust::device_vector<Vector3f> d_pointsVec(size*3);
		thrust::host_vector<Bbox3f> h_boxesVec(size);
		thrust::device_vector<Bbox3f> d_boxesVec = h_boxesVec;
		thrust::device_vector<Vector3f> d_centersVec(size,nih::Vector3f(0.f));

		for(int i=0; i<objSize; i++)
		{
			Vector3f* d_points = thrust::raw_pointer_cast(&d_pointsVec[offsets[i]*3]);
			Vector3f* d_centers = thrust::raw_pointer_cast(&d_centersVec[offsets[i]]);
			Bbox3f* d_boxes = thrust::raw_pointer_cast(&d_boxesVec[offsets[i]]);

			LoadOBJReader(objs[i],matrixes,d_points,d_centers,d_boxes);
		}
		delete[] offsets;
		/*thrust::host_vector<Vector3f> h_points(d_centersVec);
		for(int i=0; i<h_points.size(); i++)
		std::cout<<h_points[i][0]<<","<<h_points[i][1]<<","<<h_points[i][2]<<std::endl;
		h_points = (d_pointsVec);
		std::cout<<"-------\n";
		for(int i=0; i<h_points.size(); i++)
		std::cout<<h_points[i][0]<<","<<h_points[i][1]<<","<<h_points[i][2]<<std::endl;*/

		//������Χ��
		nih::Bbox3f h_gBox = thrust::reduce(d_boxesVec.begin(),d_boxesVec.end(),nih::Bbox3f(),Add_Bbox<nih::Vector3f>());

		thrust::device_vector<Bvh_Node> nodes(size-1);
		thrust::device_vector<Bvh_Node> leaves(size);
		cub::CachingDeviceAllocator allocator(true);
		KBvh_Builder builder(nodes,leaves,allocator);
		cuda::DBVH h_bvh;
		builder.build(h_gBox,d_centersVec.begin(),d_centersVec.end(),d_boxesVec.begin(),d_boxesVec.end(),&h_bvh);

		thrust::host_vector<Bvh_Node> h_nodes(nodes);
		thrust::host_vector<Bvh_Node> h_leaves(leaves);	
		thrust::host_vector<Bbox3f> h_nodeBoxes(size-1);
		thrust::host_vector<Bbox3f> h_leafBoxes(size);

		/*
		for(int i = 0; i<h_nodes.size(); i++)
		{ 
		std::cout<<" parent idx is "<<h_nodes[i].parentIdx<<" ,";

		if(h_nodes[i].l_isleaf)
		{
		std::cout<<i<<" left child "<<" is leaf "<<h_nodes[i].getChild(0);
		}
		else
		{
		std::cout<<i<<" left child "<<" is internal "<<h_nodes[i].getChild(0);				

		}
		if(h_nodes[i].r_isleaf)
		{
		std::cout<<" right child "<<" is leaf "<<h_nodes[i].getChild(1)<<std::endl;
		}
		else
		{
		std::cout<<" right child "<<" is internal "<<h_nodes[i].getChild(1)<<std::endl;
		}
		}
		for(int i=0; i<h_leaves.size(); i++)
		{
		std::cout<<i<<" parent is "<<h_leaves[i].parentIdx<<std::endl;
		std::cout<<" pid is "<<h_leaves[i].pid<<std::endl;
		}*/





		h_nodeBoxes.resize(size-1);
		Bbox3f* p_nodeBoxes = thrust::raw_pointer_cast(&h_nodeBoxes.front());
		cudaMemcpy(p_nodeBoxes,builder.getNodeBoxes(),sizeof(Bbox3f)*(size-1),cudaMemcpyDeviceToHost);

		h_leafBoxes.resize(size);
		Bbox3f* p_leafBoxes = thrust::raw_pointer_cast(&h_leafBoxes.front());
		cudaMemcpy(p_leafBoxes,builder.getLeafBoxes(),sizeof(Bbox3f)*(size),cudaMemcpyDeviceToHost);
		//ת��ΪDFS���е���
		Bintree_node* nbvh;
		uint32 nbvh_size = size*2-1;
		nbvh = new Bintree_node[nbvh_size];	
		cuda::DFSBintree(&h_nodes,&h_leaves,&h_nodeBoxes,&h_leafBoxes,nbvh);		


		Bintree_node* d_nbvh;
		cudaMalloc((void**)&d_nbvh,sizeof(Bintree_node)*nbvh_size);
		cudaMemcpy(d_nbvh,nbvh,sizeof(Bintree_node)*nbvh_size,cudaMemcpyHostToDevice);
		cudaBindTexture( NULL, bvhTex,
			d_nbvh, sizeof(Bintree_node)*nbvh_size );

		return size;
	}
	void VirtualFrustumCulling(size_t triSize,iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize)
	{
		using namespace nih;
		uint32 size = 0;
		uint32 *offsets = new uint32[objSize+1];
		Vector3f veye(eye.GetConstDataPtr());
		Vector3f* d_eye = NULL;
		cudaMalloc((void**)&d_eye,sizeof(Vector3f));
		cudaMemcpy(d_eye,&veye,sizeof(Vector3f),cudaMemcpyHostToDevice);
		for(int i=0; i<objSize; i++)
		{
			offsets[i] = size;
			size += objs[i]->GetTriangleCount();

		}
		thrust::device_vector<Vector3f> d_pointsVec(size*3);
		thrust::device_vector<TriFrustum> d_frustumVec(size);
		thrust::host_vector<Vector3f> h_pointsVec(size*3);
		
		for( int i=0; i<objSize; i++)
		{
			std::vector<iglu::vec3> vertices = objs[i]->GetVertecies();
			std::vector<iglu::IGLUOBJTri*> triangles = objs[i]->GetTriangles();
			uint32 offset = offsets[i]*3;
			for(int j=0; j<triangles.size(); j++)
			{			
				for( int k=0; k<3; k++)
				{
					Vector3f p(vertices[triangles[i]->vIdx[k]].GetConstDataPtr());
					h_pointsVec[offset++] = p;				
				}
			}
		}
		d_pointsVec = h_pointsVec;
		size_t n_blocks = GridSize(size);
		Vector3f* d_p123 = thrust::raw_pointer_cast(&d_pointsVec.front());
		TriFrustum* d_frustums = thrust::raw_pointer_cast(&d_frustumVec.front());
		GenerateVirFrustumKernel<<<n_blocks,128>>>(d_eye,d_p123,d_frustums,farD, size);
		thrust::device_vector<TriFrustum>::iterator values_end = thrust::remove_if(d_frustumVec.begin(),d_frustumVec.end(),is_frustum());
		// since the values after values_end are garbage, we'll resize the vector
		d_frustumVec.resize(values_end - d_frustumVec.begin());

		//culling 
		size_t frustumSize = d_frustumVec.size();
		thrust::device_vector<cullingContext> d_vectorf(triSize * frustumSize);
		cullingContext* d_list = thrust::raw_pointer_cast(&d_vectorf.front());
		n_blocks = GridSize(frustumSize);
		TriFrustum* d_tfrustumPtr = thrust::raw_pointer_cast(&d_frustumVec.front());
		FrustumCullingKernel<<<n_blocks,128>>>(d_tfrustumPtr,d_frustumVec.size(), triSize,d_list);
		
		thrust::count_if(thrust::device,d_vectorf.begin(),d_vectorf.end(),is_valid());		
		thrust::device_vector<cullingContext>fresult;
		thrust::copy_if(d_vectorf.begin(),d_vectorf.end(),fresult.begin(),is_valid());

		cudaFree(d_eye);
		delete[] offsets;
	}
	void GenVirtualFrustums(iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize)
	{
		using namespace nih;
		uint32 size = 0;
		uint32 *offsets = new uint32[objSize+1];
		Vector3f veye(eye.GetConstDataPtr());
		Vector3f* d_eye = NULL;
		cudaMalloc((void**)&d_eye,sizeof(Vector3f));
		cudaMemcpy(d_eye,&veye,sizeof(Vector3f),cudaMemcpyHostToDevice);
		for(int i=0; i<objSize; i++)
		{
			offsets[i] = size;
			size += objs[i]->GetTriangleCount();

		}
		thrust::device_vector<Vector3f> d_pointsVec(size*3);
		thrust::device_vector<TriFrustum> d_frustumVec(size);
		thrust::host_vector<Vector3f> h_pointsVec(size*3);
		
		for( int i=0; i<objSize; i++)
		{
			std::vector<iglu::vec3> vertices = objs[i]->GetVertecies();
			std::vector<iglu::IGLUOBJTri*> triangles = objs[i]->GetTriangles();
			uint32 offset = offsets[i]*3;
			for(int j=0; j<triangles.size(); j++)
			{			
				for( int k=0; k<3; k++)
				{
					Vector3f p(vertices[triangles[i]->vIdx[k]].GetConstDataPtr());
					h_pointsVec[offset++] = p;				
				}
			}
		}
		d_pointsVec = h_pointsVec;
		size_t n_blocks = GridSize(size);
		Vector3f* d_p123 = thrust::raw_pointer_cast(&d_pointsVec.front());
		TriFrustum* d_frustums = thrust::raw_pointer_cast(&d_frustumVec.front());
		GenerateVirFrustumKernel<<<n_blocks,128>>>(d_eye,d_p123,d_frustums,farD, size);
		thrust::device_vector<TriFrustum>::iterator values_end = thrust::remove_if(d_frustumVec.begin(),d_frustumVec.end(),is_frustum());
		// since the values after values_end are garbage, we'll resize the vector
		d_frustumVec.resize(values_end - d_frustumVec.begin());

		cudaFree(d_eye);
		delete[] offsets;
	}
}