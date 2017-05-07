#ifndef RANDOMNUMBERGENERATORIMPL_H
#define RANDOMNUMBERGENERATORIMPL_H

#include "global/NumberGenerator.h"

class NumberGeneratorImpl
	: public NumberGenerator
{
public:
	NumberGeneratorImpl(QObject* parent = nullptr);
	virtual ~NumberGeneratorImpl() = default;

	virtual void init(std::uint32_t arraySize, std::uint16_t threadId) override;

	virtual quint32 getRandomInt() override;
	virtual quint32 getRandomInt(quint32 range) override;
	virtual quint32 getLargeRandomInt(quint32 range) override;
	virtual qreal getRandomReal(qreal min, qreal max) override;
	virtual qreal getRandomReal() override;

	virtual quint64 getTag() override;

private:
	quint32 getNumberFromArray();

	int _index = 0;
	std::vector<std::uint32_t> _arrayOfRandomNumbers;
	std::uint64_t _runningNumber = 0;
	std::uint64_t _threadId = 0;
};

#endif // RANDOMNUMBERGENERATORIMPL_H
