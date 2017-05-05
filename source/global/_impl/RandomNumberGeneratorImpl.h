#ifndef RANDOMNUMBERGENERATORIMPL_H
#define RANDOMNUMBERGENERATORIMPL_H

#include "global/RandomNumberGenerator.h"

class RandomNumberGeneratorImpl
	: public RandomNumberGenerator
{
public:
	RandomNumberGeneratorImpl(int arraySize, QObject* parent = nullptr);
	virtual ~RandomNumberGeneratorImpl() = default;

	virtual void setSeed(quint32 value) override;

	virtual quint32 getInt() override;
	virtual quint32 getInt(quint32 range) override;
	virtual quint32 getLargeInt(quint32 range) override;
	virtual qreal getReal(qreal min, qreal max) override;
	virtual qreal getReal() override;

private:
	quint32 getNumberFromArray();

	int _index = 0;
	std::vector<quint32> _arrayOfRandomNumbers;
};

#endif // RANDOMNUMBERGENERATORIMPL_H
