#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "SceneHelper.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> 
#define MAXLINELENGTH 255
using namespace std;


//code grab from cw
/* takes an entire string, and makes it all lowercase */
void MakeLower( char *buf )
{
  char *tmp = buf;

  while ( tmp[0] != 0 )
    {
      *tmp = (char)tolower( *tmp );
      tmp++;
    }
}

/* takes an entire string, and makes it all uppercase */
void MakeUpper( char *buf )
{
  char *tmp = buf;

  while ( tmp[0] != 0 )
    {
      *tmp = (char)toupper( *tmp );
      tmp++;
    }
}

/* check to see if a character is a white space. */
/*    NOTE: no functions in this file use this!  */
int IsWhiteSpace( char c )
{
	if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 1;
	return 0;
}

/* 
** Returns a ptr to the first non-whitespace character 
** in a string
*/
char *StripLeadingWhiteSpace( char *string )
{
  char *tmp = string;

  while ( (tmp[0] == ' ') ||
	  (tmp[0] == '\t') ||
	  (tmp[0] == '\n') ||
	  (tmp[0] == '\r') )
    tmp++;

  return tmp;
}

/* 
** Returns a ptr to the first non-whitespace character 
** in a string...  Also includes the special character(s) 
** when it considers whitespace (useful for stripping
** off commas, parenthesis, etc)
*/
char *StripLeadingSpecialWhiteSpace( char *string, char special, char special2 )
{
  char *tmp = string;

  while ( (tmp[0] == ' ') ||
	  (tmp[0] == '\t') ||
	  (tmp[0] == '\n') ||
	  (tmp[0] == '\r') ||
	  (tmp[0] == special) ||
	  (tmp[0] == special2) )
    tmp++;
  
  return tmp;
}

/*
** Returns the first 'token' (string of non-whitespace
** characters) in the buffer, and returns a pointer to the
** next non-whitespace character (if any) in the string
*/
char *StripLeadingTokenToBuffer( char *string, char *buf )
{
  char *tmp = string;
  char *out = buf;

  while ( (tmp[0] != ' ') &&
	  (tmp[0] != '\t') &&
	  (tmp[0] != '\n') &&
	  (tmp[0] != '\r') &&
	  (tmp[0] != 0) )
    {
      *(out++) = *(tmp++); 
    }
  *out = 0;

  return StripLeadingWhiteSpace( tmp );
}


/*
** Returns the first 'token' (string of non-whitespace
** characters) in the buffer, and returns a pointer to the
** next non-whitespace character (if any) in the string
**
** Also, this function stops if it encounters either of the special
** characters passed in by the user, and returns the token thus far...
** This is useful for reading till a comma or a parenthesis, etc
**
** Returns the reason it stopped (as a character) if passed a pointer
** to a character.
*/
char *StripLeadingSpecialTokenToBuffer ( char *string, char *buf, 
					 char special, char special2, char *reason)
{
  char *tmp = string;
  char *out = buf;

  while ( (tmp[0] != ' ') &&
	  (tmp[0] != '\t') &&
	  (tmp[0] != '\n') &&
	  (tmp[0] != '\r') &&
	  (tmp[0] != 0) &&
	  (tmp[0] != special) &&
	  (tmp[0] != special2) )
    {
      *(out++) = *(tmp++); 
    }
  
  if (reason) *reason = tmp[0];
  *out = 0;

  return StripLeadingSpecialWhiteSpace( tmp, tmp[0], 0 );
}

/*
** works the same ways as StripLeadingTokenToBuffer,
** except instead of returning a string, it returns
** basically atof( StripLeadingTokenToBuffer ), with
** some minor cleaning beforehand to make sure commas,
** parens and other junk don't interfere.
*/
char *StripLeadingNumber( char *string, float *result )
{
  char *tmp = string;
  char buf[80];
  char *ptr = buf;
  char *ptr2;

  tmp = StripLeadingTokenToBuffer( tmp, buf );
  
  /* find the beginning of the number */
  while( (ptr[0] != '-') &&
	 (ptr[0] != '.') &&
	 ((ptr[0]-'0' < 0) ||
	  (ptr[0]-'9' > 0)) )
    ptr++;

  /* find the end of the number */
  ptr2 = ptr;
  while( (ptr2[0] == '-') ||
	 (ptr2[0] == '.') ||
	 ((ptr2[0]-'0' >= 0) && (ptr2[0]-'9' <= 0)) )
    ptr2++;

  /* put a null at the end of the number */
  ptr2[0] = 0;

  *result = (float)atof(ptr);

  return tmp;
}

glm::vec3 parseVec3(char* buffer)
{
	glm::vec3 result;
	char *ptr;
	ptr = StripLeadingNumber( buffer, &result[0] );
	ptr = StripLeadingNumber( ptr, &result[1] );
	ptr = StripLeadingNumber( ptr, &result[2] );
	return result;
}
SceneData::SceneData(const char* fileName)
{
	FILE *sceneFile = fopen( fileName, "r" );
	if (!sceneFile) 
	{
		printf( "Scene::Scene() unable to open scene file '%s'!\n", fileName );
		return;
	}
	
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, sceneFile) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr == 0 || ptr[0] == '\n' || ptr[0] == '#') continue;
		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.  You will need to add more!
		if (!strcmp(token, "")) continue;
		else if (!strcmp(token,"camera"))  
			 loadCamera( ptr, sceneFile);
		else if (!strcmp(token,"light")) 
			 loadLights(ptr,sceneFile);
		else if (!strcmp(token,"object"))		
			 loadObjModel( ptr, sceneFile );		
		else
			printf( "Unknown scene command '%s' in Scene::Scene()!\n", token );
	}
}

void SceneData::loadCamera(char* typeString, FILE* file)
{
	char type[10];
	char* ptr = StripLeadingTokenToBuffer( typeString, type );
	if (!strcmp(type,"pinhole"))
	{
		_camera = CameraData(file);
		_camera.pinhole = true;
	}
	else
	{
		_camera = CameraData(file);
		_camera.pinhole = false;
	}
	
}
void SceneData::loadLights(char* type, FILE* file)
{
	Light nl(file);
	_lights.push_back(nl);
}

void SceneData::loadObjModel(char* linePtr, FILE*file)
{
	char token[256], *ptr;
	ptr = StripLeadingTokenToBuffer( linePtr, token );

	SceneObject* objPtr = NULL;
	//if (!strcmp(token,"sphere"))
	//	//UnhandledKeyword( file, "object type", token );
	//	objPtr = new Sphere( file, this );
	//else if (!strcmp(token,"parallelogram"))
	//	objPtr = new Quad( file, this );
	//else if (!strcmp(token,"texparallelogram"))
	//	objPtr = new Quad( file, this );
	//else if (!strcmp(token,"testquad"))
	//	objPtr = new Quad( file, this );
	//else if (!strcmp(token,"testdisplacedquad"))
	//	objPtr = new Quad( file, this );
	//else if (!strcmp(token,"noisyquad"))
	//	objPtr = new Quad( file, this );
	//else if (!strcmp(token,"triangle") || !strcmp(token,"tri"))
	//	objPtr = new Triangle( file, this );
	//else if (!strcmp(token,"textri"))
	//	objPtr = new Triangle( file, this );
	//else if (!strcmp(token,"cyl") || !strcmp(token,"cylinder"))
	//	objPtr = new Cylinder( file, this );
	if (!strcmp(token,"mesh"))
	{	
		ptr = StripLeadingTokenToBuffer( ptr, token );
		if (!strcmp(token,"obj"))
		{
			objPtr = new ObjModelObject(ptr, file );
		}
		
	}
	else if(!strcmp(token,"plane"))
			objPtr = new PlaneObject(ptr,file);
	/*else if (!strcmp(token,"group"))
		objPtr = new GroupObject( ptr, file );*/
	else
		printf("Unknown object type '%s' in LoadObject()!\n", token);

	if (objPtr)
	{
		_objs.push_back(objPtr);
	}
}


CameraData::CameraData(FILE* f)
{
	// Search the scene file.
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );

		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		if (!strcmp(token,"eye")) 
			eye = parseVec3(ptr);
		else if (!strcmp(token,"at")) 
			at = parseVec3(ptr);
		else if (!strcmp(token,"up")) 
			up = parseVec3(ptr);
		/*else if (!strcmp(token,"w") || !strcmp(token,"width") )  
		{
		ptr = StripLeadingTokenToBuffer( ptr, token );
		s->SetWidth( (int)atof( token ) );
		}
		else if (!strcmp(token,"h") || !strcmp(token,"height") )  
		{
		ptr = StripLeadingTokenToBuffer( ptr, token );
		s->SetHeight( (int)atof( token ) );
		}
		else if (!strcmp(token,"res") || !strcmp(token, "resolution"))
		{
		ptr = StripLeadingTokenToBuffer( ptr, token );
		s->SetWidth( (int)atof( token ) );
		ptr = StripLeadingTokenToBuffer( ptr, token );
		s->SetHeight( (int)atof( token ) );
		}*/
		/*else if (!strcmp(token, "matrix"))
		{
		if (!mat) 
		mat = new Matrix4x4( f, ptr );
		else
		(*mat) *= Matrix4x4( f, ptr );
		}			*/
		else if (!strcmp(token,"fovy"))  
			ptr = StripLeadingNumber( ptr, &fovy );
		else if (!strcmp(token,"near"))  
			ptr = StripLeadingNumber( ptr, &zNear );
		else if (!strcmp(token,"far"))  
			ptr = StripLeadingNumber( ptr, &zFar );			
		else
			printf("Unknown command '%s' when loading Camera!\n", token);
	}
}

Light::Light(FILE* f)
{
	//默认产生阴影
	castShadow = 1;

	// Search the scene file.
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );

		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		if (!strcmp(token,"pos")) 
			pos = parseVec3(ptr);
		else if (!strcmp(token,"at")) 
			at = parseVec3(ptr);
		else if (!strcmp(token,"color")) 
			color = parseVec3(ptr);	
		else if (!strcmp(token,"shadow")) 
		{
			float shadow = 1;
			ptr = StripLeadingNumber(ptr,&shadow);
			castShadow = shadow > 0 ? 1 : 0;
		}
		else
			printf("Unknown command '%s' when loading Light!\n", token);
	}
}

ObjModelObject::ObjModelObject(char* name, FILE* f)
{

	_type = SceneObjType::mesh;

	char buf[ MAXLINELENGTH ], token[256], *ptr;
	ptr = StripLeadingTokenToBuffer( name, token );
	_objName = strdup(token);
	// Now find out the other model parameters
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		else if (!strcmp(token,"file"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			_objFileName = strdup(token);
		}
		else if(!strcmp(token,"material"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			_materialName = strdup(token);
		}
		else if (!strcmp(token,"matrix"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			MakeLower( token );
			if (!strcmp(token,"translate"))
			{
				glm::vec3 tran = parseVec3(ptr);
				_transform *= glm::translate(glm::mat4(1.0),tran);
			}
			else if(!strcmp(token,"rotate"))
			{
				float ang = 0;
				ptr = StripLeadingNumber(ptr,&ang);
				glm::vec3 axis = parseVec3(ptr);
				_transform *= glm::rotate(glm::mat4(1.0),ang,axis);
			}
			else if(!strcmp(token,"scale"))
			{
				glm::vec3 scale = parseVec3(ptr);
				_transform *= glm::scale(glm::mat4(1.f),scale);
			}
		}
			
	}
}

PlaneObject::PlaneObject(char* name, FILE* f)
{
	_type = SceneObjType::parallelogram;
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	
	_objName = strdup(name);
	// Now find out the other model parameters
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		else if (!strcmp(token,"anchor"))
		{
			_anchor = parseVec3(ptr);
		}
		else if(!strcmp(token,"material"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			_materialName = strdup(token);
		}
		else if (!strcmp(token,"v1"))
		{
			_v1 = parseVec3(ptr);
		}
		else if (!strcmp(token,"v2"))
		{
			_v2 = parseVec3(ptr);
		}
		else if (!strcmp(token,"matrix"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			MakeLower( token );
			if (!strcmp(token,"translate"))
			{
				glm::vec3 tran = parseVec3(ptr);
				_transform *= glm::translate(glm::mat4(1.0),tran);
			}
			else if(!strcmp(token,"rotate"))
			{
				float ang = 0;
				ptr = StripLeadingNumber(ptr,&ang);
				glm::vec3 axis = parseVec3(ptr);
				_transform *= glm::rotate(glm::mat4(1.0),ang,axis);
			}
			else if(!strcmp(token,"scale"))
			{
				glm::vec3 scale = parseVec3(ptr);
				_transform *= glm::scale(glm::mat4(1.f),scale);
			}
		}
			
	}
}