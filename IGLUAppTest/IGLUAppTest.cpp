#include "IGLUApp.h"
IGLUApp * app;
void main()
{

	app= new IGLUApp("../../CommonSampleFiles/scenes/cityIsland.txt");	
	app->Run();
	
}