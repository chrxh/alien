#include "NumberGenerator.h"

#include <random>
#include <sstream>
#include <thread>

#include <Base/Definitions.h>

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

uint64_t NumberGenerator::createEntityId()
{
    return _ids.entityId++;
}

uint64_t NumberGenerator::createLineageId()
{
    return _ids.lineageId++;
}

void NumberGenerator::adaptMaxEntityId(uint64_t id)
{
    _ids.entityId = std::max(_ids.entityId, id + 1);
}

void NumberGenerator::adaptMaxLineageId(uint32_t id)
{
    _ids.lineageId = std::max(_ids.lineageId, id + 1);
}

void NumberGenerator::adaptMaxIds(Ids const& ids)
{
    adaptMaxEntityId(ids.entityId);
}

void NumberGenerator::setIds(Ids const& ids)
{
    _ids = ids;
}

double NumberGenerator::getRandomDouble(double min, double max)
{
    return getLargeRandomInt(static_cast<int>((max - min) * 1000)) / 1000.0 + min;
}

float NumberGenerator::getRandomFloat(float min, float max)
{
    return toFloat(getRandomDouble(min, max));
}

double NumberGenerator::getRandomDouble()
{
    return static_cast<double>(getNumberFromArray()) / static_cast<double>(std::numeric_limits<int>::max());
}

uint32_t NumberGenerator::getNumberFromArray()
{
    _currentRandomNumberIndex = (_currentRandomNumberIndex + 1) % _arrayOfRandomNumbers.size();
    return _arrayOfRandomNumbers[_currentRandomNumberIndex];
}
