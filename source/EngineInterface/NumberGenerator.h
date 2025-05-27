#pragma once

#include "Base/Singleton.h"

#include "Ids.h"

class NumberGenerator
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NumberGenerator);

public:

	uint32_t getRandomInt();
    uint32_t getRandomInt(uint32_t range);
    uint32_t getRandomInt(uint32_t min, uint32_t max);
    double getRandomReal();
    double getRandomReal(double min, double max);
    float getRandomFloat(float min, float max);
    uint32_t getLargeRandomInt(uint32_t range);

    uint64_t createObjectId();
    uint64_t createCreatureId();

    void adaptMaxIds(Ids const& ids);

private:
    NumberGenerator();

    uint32_t getNumberFromArray();

	int _currentRandomNumberIndex = 0;
	std::vector<uint32_t> _arrayOfRandomNumbers;

    Ids _ids;
};

