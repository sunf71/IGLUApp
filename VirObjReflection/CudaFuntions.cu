#include "CudaFunctions.h"
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include "BVH\cuda_klbvh.h"
#include "BVH\cuda_klbvh.cuh"
#include <iostream>
#include <fstream>
#include "BVH/GpuTimer.cuh"
#include <cub/cub.cuh>

texture<uint32> bvhTex;
//创建BVH时排序后的图元序列
texture<uint32> indexTex;
//裁剪后的虚物体，注意纹理数据需要保存在全局内存中
texture<uint32> virObjTex;

cullingContext* cullingResult;
//创建bvh时排序后的图元序号
uint32* gd_indices;

__global__ void GenerateVirtualFrustumKernel(Vector3f* eye,Vector3f* p123, uint32* marixId, float* matrix,TriFrustum* frustums,unsigned* Ids , float farD, int count);
	

namespace cuda
{
	struct is_valid_frustum
	{
		__forceinline__ __host__ __device__ bool operator()(const TriFrustum& c) const
		{
			return c.id != unsigned(-1);
		}
	};

	struct bvhTexHelper
	{
		static const uint32 nodeSize = 11;
		static const uint32 LChildOf = 6;
		static const uint32 RChildOf = 7;
		static const uint32 pidOf = 8;
		static const uint32 leafStartOf = 9;
		static const uint32 leafEndOf = 10;
		float p[6];
		NIH_DEVICE float* getBbox(uint32 id)
		{
			uint32 offset = id*nodeSize;
			uint32 t[6];

			for(int i=0; i<6; i++)
			{
				t[i] = tex1Dfetch(bvhTex,offset+i);
				p[i] = bitsToFloat(t[i]);
			}


			return p;
		}

		NIH_DEVICE uint32 getLChild(uint32 id)
		{
			return tex1Dfetch(bvhTex,id*nodeSize+LChildOf);
		}

		NIH_DEVICE uint32 getRChild(uint32 id)
		{
			return tex1Dfetch(bvhTex,id*nodeSize+RChildOf);
		}
		NIH_DEVICE uint32 getPid(uint32 id)
		{
			return tex1Dfetch(bvhTex,id*nodeSize+pidOf);
		}
		NIH_DEVICE uint32 getleafStart(uint32 id)
		{
			return tex1Dfetch(bvhTex,id*nodeSize+leafStartOf);
		}
		NIH_DEVICE uint32 getleafEnd(uint32 id)
		{
			return tex1Dfetch(bvhTex,id*nodeSize+leafEndOf);
		}
		NIH_DEVICE bool isLeaf(uint32 id)
		{
			return getleafStart(id) == getleafEnd(id);
		}
	};

	FORCE_INLINE NIH_DEVICE void FrustumCulling(TriFrustum& frustum, uint32 frustumId, uint32 idx,
		uint32 priSize,
		cullingContext* out)
	{
		bvhTexHelper helper;
		uint32 offset = priSize*idx;
		const uint32 stack_size  = 64;
		uint32 stack[stack_size];
		uint32 top = 0;
		stack[top++] = 0;
		while(top>0)
		{
			uint32 idx = stack[--top];
			//Bintree_node * node = &bvh[idx];
			uint32 RChild = helper.getRChild(idx);
			uint32 LChild = helper.getLChild(idx);

			int ret = Intersect(frustum,helper.getBbox(idx));
			if (ret == 2)
			{
				//相交
				if(helper.isLeaf(RChild))
				{
					if (Intersect(frustum,helper.getBbox(RChild)))
					{
						uint32 pid = helper.getPid(RChild);
						pid = tex1Dfetch(indexTex,pid);
						out[offset+pid].frustumId = frustumId;
						out[offset+pid].triId = pid;					
					}				
				}
				else
					stack[top++] = RChild;

				if (helper.isLeaf(LChild))
				{

					if (Intersect(frustum,helper.getBbox(LChild)))
					{
						uint32 pid = helper.getPid(LChild);
						pid = tex1Dfetch(indexTex,pid);
						out[offset+pid].frustumId = frustumId;					
						out[offset+pid].triId = pid;
					}				
				}
				else
					stack[top++] = LChild;
			}
			else if (ret == 1)
			{
				//in
				for(int k= helper.getleafStart(idx); k<=helper.getleafEnd(idx);k++)
				{						
					uint32	pid = tex1Dfetch(indexTex,k);
					out[offset+pid].frustumId = frustumId;
					out[offset+pid].triId = pid;
				}
			}
		}
	}

	

	__global__ void FrustumCullingKernel(TriFrustum* frustumP, int frustum_num, uint32 priSize,cullingContext* list)
	{
		uint32 step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < frustum_num; 
			i += step) 
		{
			TriFrustum frustum = frustumP[i];
			FrustumCulling(frustum,frustumP[i].id,i,priSize,list);
			//FrustumCullingT(frustum,i,bvh,priSize,list);
		}
	}
	

	size_t GridSize(size_t jobSize,size_t blockSize = 128)
	{

		int numSMs;
		cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		size_t max_blocks = 65535;
		return numSMs*nih::min( max_blocks, (jobSize + (blockSize*numSMs)-1) / (blockSize*numSMs) );
	}

	//@vertices 是顶点数组，每个顶点用3个浮点数表示
	//@indices 是三角形顶点的索引数组，每个三角形有3个顶点
	//@matrix 是4×4模型矩阵
	//@points 输出参数，所有三角形顶点
	//@centers 输出参数，所有三角形的中心
	//@boxes 输出参数，所有三角形的包围盒
	//@size 三角形个数
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
			centers[i] = MatrixXVector3f(matrix,centers[i]);
			boxes[i].m_min = MatrixXVector3f(matrix,boxes[i].m_min);
			boxes[i].m_max = MatrixXVector3f(matrix,boxes[i].m_max);
		}
	}

	//根据原有的索引和裁剪结果生成新的索引
	__global__ void UpdateElementKernel(const unsigned* inPtr, unsigned * outPtr,  cullingContext* cullingResult, int size)
	{
		unsigned step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < size; 
			i += step) 
		{
			
			unsigned offset = i*3;
			unsigned offset2 = tex1Dfetch(virObjTex,i*2+1)*3;
			//unsigned offset2 = cullingResult[i].triId*3;
			outPtr[offset] = inPtr[offset2];
			outPtr[offset+1] = inPtr[offset2+1];
			outPtr[offset+2] = inPtr[offset2+2];
		}
	}



	__global__ void UpdateCommandKernel(float* vAttrPtr, unsigned* vCmdPtr,unsigned size)
	{
		unsigned step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < size; 
			i += step) 
		{
			
			unsigned offset = i*5;
			unsigned offset2 = tex1Dfetch(virObjTex,i*2+1)*3;		
			unsigned offset3 = i*3;
			vCmdPtr[offset] = 3;
			vCmdPtr[offset+1] = 1;
			vCmdPtr[offset+2] = offset2;
			vCmdPtr[offset+3] = 0;
			vCmdPtr[offset+4] =i;
			vAttrPtr[i] = tex1Dfetch(virObjTex,i*2);
		}
	}
	__global__  void UpdateMirrorElementKernel(const unsigned* mirrorId, const unsigned* inPtr, unsigned* outPtr,unsigned size)
	{
		unsigned step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < size; 
			i += step) 
		{
			unsigned offset = 3*i;
			unsigned offset2 = mirrorId[i]*3;
			outPtr[offset] = inPtr[offset2];
			outPtr[offset+1] = inPtr[offset2+1];
			outPtr[offset+2] = inPtr[offset2 +2];
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
		/*for(int i=0; i<size*3; i+=3)
		std::cout<<indices[i]<<","<<indices[i+1]<<","<<indices[i+2]<<std::endl;*/
		float* d_vertices;
		cudaMalloc((void**)&d_vertices,sizeof(float)*vertSize);
		cudaMemcpy(d_vertices,vertices,sizeof(float)*vertSize,cudaMemcpyHostToDevice);

		uint32* d_indices;
		cudaMalloc((void**)&d_indices,sizeof(uint32)*3*size);
		cudaMemcpy(d_indices,indices,sizeof(uint32)*3*size,cudaMemcpyHostToDevice);

		float* d_matrix;
		cudaMalloc((void**)&d_matrix,sizeof(float)*16);
		cudaMemcpy(d_matrix,matrix->GetConstDataPtr(),sizeof(float)*16,cudaMemcpyHostToDevice);

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
		LoadOBJKernel<<<n_blocks,128>>>(d_vertices,d_indices,d_matrix,d_points,d_centers,d_boxes,size);
		cudaFree(d_indices);
		cudaFree(d_vertices);
		cudaFree(d_matrix);
	}

	//
	size_t BuildBvh(iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize)
	{
		using namespace nih;
		GpuTimer timer;
		
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

			LoadOBJReader(objs[i],matrixes+i,d_points,d_centers,d_boxes);
		}
		delete[] offsets;
	/*	thrust::host_vector<Vector3f> h_points(d_centersVec);
		for(int i=0; i<h_points.size(); i++)
		std::cout<<h_points[i][0]<<","<<h_points[i][1]<<","<<h_points[i][2]<<std::endl;
		h_points = (d_pointsVec);
		std::cout<<"-------\n";
		for(int i=0; i<h_points.size(); i++)
		std::cout<<h_points[i][0]<<","<<h_points[i][1]<<","<<h_points[i][2]<<std::endl;*/

		//求最大包围盒
		nih::Bbox3f h_gBox = thrust::reduce(d_boxesVec.begin(),d_boxesVec.end(),nih::Bbox3f(),Add_Bbox<nih::Vector3f>());
		
		/*std::cout<<"输出载入obj顶点数据"<<std::endl;
		thrust::host_vector<nih::Vector3f> h_pointsVec(d_pointsVec);
		for(int i=0; i<h_pointsVec.size();i+=3)
		{
			std::cout<<i<<std::endl;
			for(int j=0;j<3;j++)
				std::cout<<h_pointsVec[i+j][0]<<","<<h_pointsVec[i+j][1]<<","<<h_pointsVec[i+j][2]<<std::endl;
		}*/


		thrust::device_vector<Bvh_Node> nodes(size-1);
		thrust::device_vector<Bvh_Node> leaves(size);
		cub::CachingDeviceAllocator allocator(true);
		KBvh_Builder builder(nodes,leaves,allocator);
		timer.Start();
		cuda::DBVH h_bvh;
		builder.build(h_gBox,d_centersVec.begin(),d_centersVec.end(),d_boxesVec.begin(),d_boxesVec.end(),&h_bvh);
		timer.Stop();
		timer.Report("Building bvh");
		thrust::host_vector<Bvh_Node> h_nodes(nodes);
		thrust::host_vector<Bvh_Node> h_leaves(leaves);	
		thrust::host_vector<Bbox3f> h_nodeBoxes(size-1);
		thrust::host_vector<Bbox3f> h_leafBoxes(size);

		//
		//for(int i = 0; i<h_nodes.size(); i++)
		//{ 
		//std::cout<<" parent idx is "<<h_nodes[i].parentIdx<<" ,";

		//if(h_nodes[i].l_isleaf)
		//{
		//std::cout<<i<<" left child "<<" is leaf "<<h_nodes[i].getChild(0);
		//}
		//else
		//{
		//std::cout<<i<<" left child "<<" is internal "<<h_nodes[i].getChild(0);				

		//}
		//if(h_nodes[i].r_isleaf)
		//{
		//std::cout<<" right child "<<" is leaf "<<h_nodes[i].getChild(1)<<std::endl;
		//}
		//else
		//{
		//std::cout<<" right child "<<" is internal "<<h_nodes[i].getChild(1)<<std::endl;
		//}
		//}
		//for(int i=0; i<h_leaves.size(); i++)
		//{
		//std::cout<<i<<" parent is "<<h_leaves[i].parentIdx<<std::endl;
		//std::cout<<" pid is "<<h_leaves[i].pid<<std::endl;
		//}

		h_nodeBoxes.resize(size-1);
		Bbox3f* p_nodeBoxes = thrust::raw_pointer_cast(&h_nodeBoxes.front());
		cudaMemcpy(p_nodeBoxes,builder.getNodeBoxes(),sizeof(Bbox3f)*(size-1),cudaMemcpyDeviceToHost);

		h_leafBoxes.resize(size);
		Bbox3f* p_leafBoxes = thrust::raw_pointer_cast(&h_leafBoxes.front());
		cudaMemcpy(p_leafBoxes,builder.getLeafBoxes(),sizeof(Bbox3f)*(size),cudaMemcpyDeviceToHost);
		//转换为DFS排列的树
		Bintree_node* nbvh;
		uint32 nbvh_size = size*2-1;
		nbvh = new Bintree_node[nbvh_size];	
		cuda::DFSBintree(&h_nodes,&h_leaves,&h_nodeBoxes,&h_leafBoxes,nbvh);		


		Bintree_node* d_nbvh;
		cudaMalloc((void**)&d_nbvh,sizeof(Bintree_node)*nbvh_size);
		cudaMemcpy(d_nbvh,nbvh,sizeof(Bintree_node)*nbvh_size,cudaMemcpyHostToDevice);
		cudaBindTexture( NULL, bvhTex,
			d_nbvh, sizeof(Bintree_node)*nbvh_size );

		cudaMalloc((void**)&gd_indices,sizeof(uint32)*size);
		cudaMemcpy(gd_indices,builder.getIndices(),sizeof(uint32)*size,cudaMemcpyDeviceToDevice);
		cudaBindTexture(NULL,indexTex,gd_indices,sizeof(uint32)*size);

	
		return size;
	}
	size_t VirtualFrustumCulling(size_t triSize, size_t mirrorSize,iglu::vec3& eye, float farD, float* mirrorPos, unsigned* mirroMatrixId, iglu::IGLUMatrix4x4::Ptr matrixes, size_t& mirrorTriSize,const unsigned* mEleInPtr, unsigned * mEleOutPtr)
	{
		using namespace nih;
#ifdef TEST
		GpuTimer timer;
		timer.Start();
#endif
	
		Vector3f veye(eye.GetConstDataPtr());
		Vector3f* d_eye = NULL;
		cudaMalloc((void**)&d_eye,sizeof(Vector3f));
		cudaMemcpy(d_eye,&veye,sizeof(Vector3f),cudaMemcpyHostToDevice);
		
		thrust::device_vector<Vector3f> d_pointsVec(mirrorTriSize*3);
		thrust::device_vector<TriFrustum> d_frustumVec(mirrorTriSize);
		Vector3f* d_p123 = thrust::raw_pointer_cast(&d_pointsVec.front());
		
		cudaMemcpy(thrust::raw_pointer_cast(&d_pointsVec.front()),mirrorPos,sizeof(float)*mirrorTriSize*9,cudaMemcpyHostToDevice);
		//thrust::host_vector<Vector3f> h_pointsVec(mirrorTriSize*3);
		//h_pointsVec = d_pointsVec;
		//std::cout<<h_pointsVec[0][0]<<","<<h_pointsVec[0][1]<<","<<h_pointsVec[0][2]<<std::endl;
		//for(int i =0; i<mirrorTriSize*3;i++)
		//{
		//	//h_pointsVec[i] = Vector3f(mirrorPos+3*i);
		//	std::cout<<h_pointsVec[i][0]<<","<<h_pointsVec[i][1]<<","<<h_pointsVec[i][2]<<std::endl;
		//}
		
		size_t n_blocks = GridSize(mirrorTriSize);
		
		TriFrustum* d_frustums = thrust::raw_pointer_cast(&d_frustumVec.front());
		float* h_matrix = new float[16*mirrorSize];
		for(int i =0; i<mirrorSize; i++)
			memcpy(h_matrix+16*i,matrixes[i].GetConstDataPtr(),16*sizeof(float));
		float* d_matrix;
		cudaMalloc((void**)&d_matrix,sizeof(float)*16*mirrorSize);
		cudaMemcpy(d_matrix,h_matrix,sizeof(float)*16*mirrorSize,cudaMemcpyHostToDevice);
		uint32* d_matrixId;
		cudaMalloc((void**)&d_matrixId,sizeof(uint32)*mirrorTriSize);
		cudaMemcpy(d_matrixId,mirroMatrixId,sizeof(uint32)*mirrorTriSize,cudaMemcpyHostToDevice);

		thrust::device_vector<unsigned> d_mirrorIdVec(mirrorTriSize);
		unsigned* d_mirrorId = thrust::raw_pointer_cast(&d_mirrorIdVec.front());

		GenerateVirtualFrustumKernel<<<n_blocks,128>>>(d_eye,d_p123,d_matrixId,d_matrix,d_frustums,d_mirrorId,farD ,mirrorTriSize);
#ifdef TEST
		timer.Stop();
		timer.Report("  Generate Virtual Frustum Kernel cuda");		
#endif
		//timer.Start();
		//thrust::device_vector<TriFrustum>::iterator values_end = thrust::remove_if(d_frustumVec.begin(),d_frustumVec.end(),is_not_frustum());
		//// since the values after values_end are garbage, we'll resize the vector
		//d_frustumVec.resize(values_end - d_frustumVec.begin());
		//thrust::device_vector<unsigned>::iterator values_endPtr = thrust::remove_if(d_mirrorIdVec.begin(),d_mirrorIdVec.end(),is_negtive());
		//d_mirrorIdVec.resize(values_endPtr-d_mirrorIdVec.begin());
		///*thrust::host_vector<unsigned> h_mirrorId(d_mirrorIdVec);
		//std::cout<<"mirror ID:"<<std::endl;
		//for(int i=0; i<h_mirrorId.size(); i++)
		//	std::cout<<h_mirrorId[i]<<std::endl;
		//thrust::host_vector<cullingContext> h_fakeResult(d_mirrorIdVec.size()*10);
		//for(int i=0; i<h_fakeResult.size()/10; i++)
		//{
		//	for(int j=0; j<10; j++)
		//	{
		//		h_fakeResult[i*10+j].frustumId =h_mirrorId[i];
		//		h_fakeResult[i*10+j].triId = j;
		//	}
		//}*/
		///*thrust::host_vector<cullingContext> h_fakeResult;
		//for(int i=1; i<h_mirrorId.size(); i++)
		//{
		//	for(int j=0; j<10; j++)
		//	h_fakeResult.push_back(cullingContext(h_mirrorId[i],j));
		//}*/
		//timer.Stop();
		//timer.Report("  Generate Virtual Frustum Stream Compaction");

#ifdef TEST
		timer.Start();	
#endif
		int      num_items = mirrorTriSize;     
		TriFrustum      *d_frustumsOut;             
		cudaMalloc((void**)&d_frustumsOut,sizeof(TriFrustum)*mirrorTriSize);
		int      *d_num_selected;    
		cudaMalloc((void**)&d_num_selected,sizeof(int));
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		is_frustum op;
		cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_frustums, d_frustumsOut, d_num_selected, num_items, op);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run selection
		cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_frustums, d_frustumsOut, d_num_selected, num_items,op);
		int h_num_selected;
		cudaMemcpy(&h_num_selected,d_num_selected,sizeof(int),cudaMemcpyDeviceToHost);		
		cudaFree(d_temp_storage);
#ifdef TEST
		timer.Stop();
		timer.Report( "cub Generate Virtual Frustum Stream Compaction");
#endif

#ifdef TEST
		timer.Start();
#endif
		UpdateMirrorElement(d_mirrorId,mEleInPtr,mEleOutPtr,d_mirrorIdVec.size());
#ifdef TEST
		timer.Stop();
		timer.Report("    UpdateMirrorElement Kernel cuda");
#endif
		mirrorTriSize = d_mirrorIdVec.size();
		/*unsigned* hEle = new unsigned[d_mirrorIdVec.size()*3];
		cudaMemcpy(hEle,mEleOutPtr,d_mirrorIdVec.size()*3*sizeof(unsigned),cudaMemcpyDeviceToHost);
		dataFile<<"elements:"<<std::endl;
		for(int i=0; i<d_mirrorIdVec.size()*3; i++)
			dataFile<<hEle[i]<<std::endl;
		delete[] hEle;
		dataFile.close();*/
	/*	thrust::host_vector<unsigned> h_mid(d_mirrorIdVec);
		for(int i=0; i<h_mid.size(); i++)
			std::cout<<h_mid[i]<<std::endl;*/
		//std::cout<<"frustum size "<<d_frustumVec.size()<<std::endl;
		/*h_frustumVec = d_frustumVec;
		TriFrustum f = h_frustumVec[0];*/
		//culling 
#ifdef TEST
		timer.Start();
#endif
		size_t cullingSize = h_num_selected* triSize;
		thrust::device_vector<cullingContext> d_vectorf(cullingSize);
		cullingContext* d_list = thrust::raw_pointer_cast(&d_vectorf.front());
		n_blocks = GridSize(h_num_selected);		
		FrustumCullingKernel<<<n_blocks,128>>>(d_frustumsOut,h_num_selected, triSize,d_list);
#ifdef TEST
		timer.Stop();
		timer.Report("    Frustum Culling Kernel cuda");
#endif

		/*timer.Start();
		thrust::device_vector<cullingContext>::iterator c_end = thrust::remove_if(d_vectorf.begin(),d_vectorf.end(),is_inValid());
		int inCount = c_end - d_vectorf.begin() ;
		d_vectorf.resize(inCount);
		timer.Stop();
		timer.Report("    Frustum Culling Stream Compaction");*/

		//
#ifdef TEST
		timer.Start();	
#endif
		cullingContext * d_cullingOut;	
		cudaMalloc((void**)&d_cullingOut,sizeof(cullingContext)*cullingSize);			
		void     *d_temp_storage2 = NULL;
		size_t   temp_storage_bytes2 = 0;
		is_valid op2;
		cub::DeviceSelect::If(d_temp_storage2, temp_storage_bytes2, d_list, d_cullingOut, d_num_selected, cullingSize, op2);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage2, temp_storage_bytes2);
		//// Run selection
		cub::DeviceSelect::If(d_temp_storage2, temp_storage_bytes2, d_list, d_cullingOut, d_num_selected, cullingSize,op2);
		cudaMemcpy(&h_num_selected,d_num_selected,sizeof(int),cudaMemcpyDeviceToHost);
		cudaFree(d_num_selected);
		cudaFree(d_temp_storage2);
#ifdef TEST
		timer.Stop();
		timer.Report( "    cub  Frustum Culling Stream Compaction");
#endif
	
#ifdef TEST
		timer.Start();
#endif
		cudaMalloc((void**)&cullingResult,sizeof(cullingContext)*h_num_selected);
		cudaMemcpy(cullingResult,d_cullingOut,sizeof(cullingContext)*h_num_selected,cudaMemcpyDeviceToDevice);
		cudaBindTexture(NULL,virObjTex,cullingResult,sizeof(cullingContext)*h_num_selected);
		/*cudaMalloc((void**)&cullingResult,sizeof(cullingContext)*h_fakeResult.size());
		cudaMemcpy(cullingResult,thrust::raw_pointer_cast(&h_fakeResult.front()),sizeof(uint32)*h_fakeResult.size()*2,cudaMemcpyHostToDevice);
		cudaBindTexture(NULL,virObjTex,cullingResult,sizeof(cullingContext)*h_fakeResult.size());*/
	
		
		//n_blocks = GridSize(fresult.size());
		
	    //std::cout<<fresult.size()<<std::endl;
		/*std::ofstream dataFile("data.txt");
		thrust::host_vector<cullingContext> h_result(fresult);
		for(int i=0; i<h_result.size(); i++)
		{
			dataFile<<h_result[i].frustumId<<":"<<h_result[i].triId<<std::endl;
		}
		dataFile.close();*/
		//UpdateElementKernel<<<n_blocks,128>>>(inElemBuffer,outElemBuffer,thrust::raw_pointer_cast(&fresult.front()),fresult.size());
		cudaFree(d_eye);
		cudaFree(d_cullingOut);
		//cudaFree(gd_indices);
		
		cudaFree(d_matrix);
		cudaFree(d_matrixId);
		cudaFree(d_frustumsOut);
		delete[] h_matrix;
		//return h_fakeResult.size();
#ifdef TEST
		timer.Stop();
		timer.Report("    memory release cuda");
#endif
		return h_num_selected;
	}
	void UpdateVirtualObject(float* vAttrPtr, unsigned* vCmdPtr, unsigned size)
	{
		size_t n_blocks = GridSize(size);

		//UpdateElementKernel<<<n_blocks,128>>>(inPtr,outPtr,NULL,size);
		UpdateCommandKernel<<<n_blocks,128>>>(vAttrPtr,vCmdPtr,size);
		//float * hPtr = new float[size];
		//cudaMemcpy(hPtr,vAttrPtr,sizeof(float)*size,cudaMemcpyDeviceToHost);
		//for(int i=0; i<size; i++)
		//	std::cout<<hPtr[i]<<std::endl;
		//delete[] hPtr;
		//unsigned* cmdPtr = new unsigned[size*5];
		//cudaMemcpy(cmdPtr,vCmdPtr,sizeof(unsigned)*size*5,cudaMemcpyDeviceToHost);
		//for(int i=0; i<size*5; i+=5)
		//{
		//	std::cout<<cmdPtr[i]<<","<<cmdPtr[i+1]<<","<<cmdPtr[i+2]<<","<<cmdPtr[i+3]<<","<<cmdPtr[i+4]<<std::endl;

		//}
		cudaFree(cullingResult);
	}

	void UpdateMirrorElement(const unsigned* mirrIds,const unsigned* mCElePtr, unsigned* mElePtr, unsigned size)
	{
		size_t n_blocks = GridSize(size);
		UpdateMirrorElementKernel<<<n_blocks,128>>>(mirrIds,mCElePtr,mElePtr,size);
	}
	//void GenVirtualFrustums(iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize)
	//{
	//	using namespace nih;
	//	uint32 size = 0;
	//	uint32 *offsets = new uint32[objSize+1];
	//	Vector3f veye(eye.GetConstDataPtr());
	//	Vector3f* d_eye = NULL;
	//	cudaMalloc((void**)&d_eye,sizeof(Vector3f));
	//	cudaMemcpy(d_eye,&veye,sizeof(Vector3f),cudaMemcpyHostToDevice);
	//	for(int i=0; i<objSize; i++)
	//	{
	//		offsets[i] = size;
	//		size += objs[i]->GetTriangleCount();

	//	}
	//	thrust::device_vector<Vector3f> d_pointsVec(size*3);
	//	thrust::device_vector<TriFrustum> d_frustumVec(size);
	//	thrust::host_vector<Vector3f> h_pointsVec(size*3);

	//	for( int i=0; i<objSize; i++)
	//	{
	//		std::vector<iglu::vec3> vertices = objs[i]->GetVertecies();
	//		std::vector<iglu::IGLUOBJTri*> triangles = objs[i]->GetTriangles();
	//		uint32 offset = offsets[i]*3;
	//		for(int j=0; j<triangles.size(); j++)
	//		{			
	//			for( int k=0; k<3; k++)
	//			{
	//				Vector3f p(vertices[triangles[i]->vIdx[k]].GetConstDataPtr());
	//				h_pointsVec[offset++] = p;				
	//			}
	//		}
	//	}
	//	d_pointsVec = h_pointsVec;
	//	size_t n_blocks = GridSize(size);
	//	Vector3f* d_p123 = thrust::raw_pointer_cast(&d_pointsVec.front());
	//	TriFrustum* d_frustums = thrust::raw_pointer_cast(&d_frustumVec.front());
	//	GenerateVirFrustumKernel<<<n_blocks,128>>>(d_eye,d_p123,d_frustums,farD, size);
	//	thrust::device_vector<TriFrustum>::iterator values_end = thrust::remove_if(d_frustumVec.begin(),d_frustumVec.end(),is_frustum());
	//	// since the values after values_end are garbage, we'll resize the vector
	//	d_frustumVec.resize(values_end - d_frustumVec.begin());

	//	cudaFree(d_eye);
	//	delete[] offsets;
	//}
}