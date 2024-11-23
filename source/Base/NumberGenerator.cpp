#include <sstream>
#include <thread>
#include <random>

#include "NumberGenerator.h"

NumberGenerator::NumberGenerator()
{
    _arrayOfRandomNumbers.reserve(1323781);
    _runningNumber = 0;
    std::random_device rd;   //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  //Standard mersenne_twister_engine seeded with rd()
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

double NumberGenerator::getRandomReal(double min, double max)
{
	return static_cast<double>(getLargeRandomInt(static_cast<int>((max - min) * 1000)) / 1000.0 + min);
}

float NumberGenerator::getRandomFloat(float min, float max)
{
    return toFloat(getRandomReal(min, max));
}

double NumberGenerator::getRandomReal()
{
    return static_cast<double>(getNumberFromArray()) / static_cast<double>(std::numeric_limits<int>::max());
}

uint64_t NumberGenerator::getId()
{
    return (static_cast<uint64_t>(1) << 48) | ++_runningNumber; //first term is to avoid collisions with GPU-generated ids
}

uint32_t NumberGenerator::getNumberFromArray()
{
	_index = (_index + 1) % _arrayOfRandomNumbers.size();
	return _arrayOfRandomNumbers[_index];
}
