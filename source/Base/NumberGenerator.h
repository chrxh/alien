#pragma once

#include "Definitions.h"

class NumberGenerator
{
public:
    BASE_EXPORT static NumberGenerator& getInstance();

	BASE_EXPORT uint32_t getRandomInt();
    BASE_EXPORT uint32_t getRandomInt(uint32_t range);
    BASE_EXPORT uint32_t getRandomInt(uint32_t min, uint32_t max);
    BASE_EXPORT double getRandomReal();
    BASE_EXPORT double getRandomReal(double min, double max);

	BASE_EXPORT uint64_t getId();

public:
    NumberGenerator(NumberGenerator const&) = delete;
    void operator=(NumberGenerator const&) = delete;

	uint32_t getLargeRandomInt(uint32_t range);
    uint32_t getNumberFromArray();

private:
    NumberGenerator();
    ~NumberGenerator();

	int _index = 0;
	std::vector<uint32_t> _arrayOfRandomNumbers;
	uint64_t _runningNumber = 0;
	uint64_t _threadId = 0;
};

