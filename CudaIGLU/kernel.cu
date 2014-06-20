#include "GpuIndirectDrawApp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
 __global__ void UpdateCommandKernel(unsigned int* ptr,unsigned int primCount, unsigned drawCount)
{
	unsigned step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < drawCount; 
		i += step) 
	{
		unsigned offset = 5*i;
		ptr[offset] = primCount;
		ptr[offset+1] = 1;
		ptr[offset+2] = 0;
		ptr[offset+3] = 0;
		ptr[offset+4] = i;
	}
}
void UpdateDrawCommand(unsigned int* devPtr,unsigned int primCount, unsigned int size)
{
	
	const size_t BLOCK_SIZE = 128;
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	size_t max_blocks = 65535;
	size_t n_blocks   = min( max_blocks, (size + (BLOCK_SIZE*numSMs)-1) / (BLOCK_SIZE*numSMs) );

	UpdateCommandKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(devPtr,primCount,size);
	
}
iglu::IGLUApp* app;
int main()
{
    
	app = new OGL::GpuIndirectDrawApp("../../CommonSampleFiles/scenes/nature.txt");
	OGL::GpuIndirectDrawApp* GIDApp = static_cast<OGL::GpuIndirectDrawApp*>(app);
	GIDApp->SetUpdateDrawCommand(UpdateDrawCommand);

	app->Run();
    return 0;
}


