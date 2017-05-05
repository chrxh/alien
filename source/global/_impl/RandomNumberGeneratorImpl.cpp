#include "RandomNumberGeneratorImpl.h"

RandomNumberGeneratorImpl::RandomNumberGeneratorImpl(int arraySize, QObject * parent)
	: RandomNumberGenerator(parent)
{
	for (int i = 0; i < arraySize; ++i) {
		_arrayOfRandomNumbers.push_back(qrand());
	}
}

void RandomNumberGeneratorImpl::setSeed(quint32 value)
{
	_index = value % _arrayOfRandomNumbers.size();

}

quint32 RandomNumberGeneratorImpl::getInt()
{
	return getNumberFromArray();
}

quint32 RandomNumberGeneratorImpl::getInt(quint32 range)
{
	return getNumberFromArray() % range;
}

quint32 RandomNumberGeneratorImpl::getLargeInt(quint32 range)
{
	return static_cast<quint32>((static_cast<qreal>(range) * static_cast<qreal>(getNumberFromArray()) / RAND_MAX));
}

qreal RandomNumberGeneratorImpl::getReal(qreal min, qreal max)
{
	return (qreal)getLargeInt((max - min) * 1000) / 1000.0 + min;
}

qreal RandomNumberGeneratorImpl::getReal()
{
	return static_cast<qreal>(getNumberFromArray()) / RAND_MAX;
}

quint32 RandomNumberGeneratorImpl::getNumberFromArray()
{
	_index = (_index + 1) % _arrayOfRandomNumbers.size();
	return _arrayOfRandomNumbers[_index];
}
