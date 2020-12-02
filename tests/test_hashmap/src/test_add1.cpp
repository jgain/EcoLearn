#include "PlantSpatialHashmap.h"

int main(int argc, char * argv [])
{
	PlantSpatialHashmap map(10.0f, 10.0f, 90.0f, 90.0f);

	if (map.test_add())
	{
		return 0;
	}
	else
		return 1;
}
