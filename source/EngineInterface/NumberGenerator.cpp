#include <sstream>
#include <thread>
#include <random>

#include "Base/Definitions.h"

#include "NumberGenerator.h"

NumberGenerator::NumberGenerator()
{
    _arrayOfRandomNumbers.reserve(1323781);
    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0);

    for (uint32_t i = 0; i < 1323781; ++i) {
        _arrayOfRandomNumbers.emplace_back(distrib(gen));
    }
}

uint32_t NumberGenerator::getRandomInt()
{
	return getNumberFromArray();
}

uint32_t NumberGenerator::getRandomInt(uint32_t range)
{
	return getNumberFromArray() % range;
}

uint32_t NumberGenerator::getRandomInt(uint32_t min, uint32_t max)
{
    auto delta = max - min + 1;
    return min + (getNumberFromArray() % delta);
}

uint32_t NumberGenerator::getLargeRandomInt(uint32_t range)
{
	return getNumberFromArray() % (range + 1);
}

uint64_t NumberGenerator::createObjectId()
{
    return _ids.currentObjectId++;
}

uint64_t NumberGenerator::createCreatureId()
{
    return _ids.currentCreatureId++;
}

void NumberGenerator::adaptMaxIds(Ids const& ids)
{
    _ids.currentObjectId = std::max(_ids.currentObjectId, ids.currentObjectId);
    _ids.currentCreatureId = std::max(_ids.currentCreatureId, ids.currentCreatureId);
    _ids.currentMutationId = std::max(_ids.currentMutationId, ids.currentMutationId);
}

double NumberGenerator::getRandomReal(double min, double max)
{
	return getLargeRandomInt(static_cast<int>((max - min) * 1000)) / 1000.0 + min;
}

float NumberGenerator::getRandomFloat(float min, float max)
{
    return toFloat(getRandomReal(min, max));
}

double NumberGenerator::getRandomReal()
{
    return static_cast<double>(getNumberFromArray()) / static_cast<double>(std::numeric_limits<int>::max());
}

uint32_t NumberGenerator::getNumberFromArray()
{
	_currentRandomNumberIndex = (_currentRandomNumberIndex + 1) % _arrayOfRandomNumbers.size();
	return _arrayOfRandomNumbers[_currentRandomNumberIndex];
}
