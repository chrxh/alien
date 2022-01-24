#pragma once

#include "Definitions.h"

class NumberGenerator
{
public:
    static NumberGenerator& getInstance();

	uint32_t getRandomInt();
    uint32_t getRandomInt(uint32_t range);
    uint32_t getRandomInt(uint32_t min, uint32_t max);
    double getRandomReal();
    double getRandomReal(double min, double max);

	uint64_t getId();

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

