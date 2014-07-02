#include "VORApp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

iglu::IGLUApp* app;
int main()
{
	using namespace OGL;
   
	app = new VORApp("../../CommonSampleFiles/scenes/virObjRef.txt");
	app->Run();
    return 0;
}


