#include "IGLUApp.h"
void main()
{

	IGLUApp::GetSingleton().InitScene("cornell.txt");
	IGLUApp::GetSingleton().Run();
	
}