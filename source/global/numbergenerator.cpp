#include "NumberGenerator.h"

namespace
{
	const int MAX_RANDOM_NUMBERS = 1234327;
}

NumberGenerator::NumberGenerator()
{
	_arrayOfRandomNumbers = new quint32[MAX_RANDOM_NUMBERS];
	for (int i = 0; i < MAX_RANDOM_NUMBERS; ++i) {
		_arrayOfRandomNumbers[i] = qrand();
	}
}

NumberGenerator::~NumberGenerator()
{
	delete[] _arrayOfRandomNumbers;
}

NumberGenerator & NumberGenerator::getInstance()
{
	static NumberGenerator instance;
	return instance;
}

void NumberGenerator::setRandomSeed(quint32 value)
{
	_currentNumber = value % MAX_RANDOM_NUMBERS;
}

quint32 NumberGenerator::random(quint32 range)
{
	return readRandomNumber() % range;
}

quint32 NumberGenerator::randomLargeNumbers (quint32 range)
{
    return static_cast<quint32>((static_cast<qreal>(range) * static_cast<qreal>(readRandomNumber()) / RAND_MAX));
}

qreal NumberGenerator::random (qreal min, qreal max)
{
    return (qreal)randomLargeNumbers((max-min)*1000)/1000.0+min;
}

qreal NumberGenerator::random()
{
	return static_cast<qreal>(readRandomNumber())/RAND_MAX;
}

quint32 NumberGenerator::readRandomNumber()
{
	_currentNumber = (_currentNumber + 1) % MAX_RANDOM_NUMBERS;
	return _arrayOfRandomNumbers[_currentNumber];
}



