#include <sstream>
#include <thread>
#include "NumberGeneratorImpl.h"


NumberGeneratorImpl::NumberGeneratorImpl(QObject * parent)
	: NumberGenerator(parent)
{
}

void NumberGeneratorImpl::init(std::uint32_t arraySize, std::uint16_t threadId)
{
	_threadId = static_cast<std::uint64_t>(threadId) << 48;
	_arrayOfRandomNumbers.clear();
	_runningNumber = 0;
	for (std::uint32_t i = 0; i < arraySize; ++i) {
		_arrayOfRandomNumbers.push_back(qrand());
	}
}

quint32 NumberGeneratorImpl::getRandomInt()
{
	return getNumberFromArray();
}

quint32 NumberGeneratorImpl::getRandomInt(quint32 range)
{
	return getNumberFromArray() % range;
}

quint32 NumberGeneratorImpl::getLargeRandomInt(quint32 range)
{
	return static_cast<quint32>((static_cast<qreal>(range) * static_cast<qreal>(getNumberFromArray()) / RAND_MAX));
}

qreal NumberGeneratorImpl::getRandomReal(qreal min, qreal max)
{
	return (qreal)getLargeRandomInt((max - min) * 1000) / 1000.0 + min;
}

qreal NumberGeneratorImpl::getRandomReal()
{
	return static_cast<qreal>(getNumberFromArray()) / RAND_MAX;
}

quint64 NumberGeneratorImpl::getTag()
{
	return _threadId | _runningNumber++;
}

quint32 NumberGeneratorImpl::getNumberFromArray()
{
	_index = (_index + 1) % _arrayOfRandomNumbers.size();
	return _arrayOfRandomNumbers[_index];
}
