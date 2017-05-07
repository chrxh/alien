#ifndef RANDOMNUMBERGENERATOR_H
#define RANDOMNUMBERGENERATOR_H

#include "Definitions.h"

class NumberGenerator
	: public QObject
{
	Q_OBJECT
public:
	NumberGenerator(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~NumberGenerator() = default;

	virtual void init(std::uint32_t arraySize, std::uint16_t threadId) = 0;

	virtual quint32 getRandomInt() = 0;
	virtual quint32 getRandomInt(quint32 range) = 0;
	virtual quint32 getLargeRandomInt(quint32 range) = 0;
	virtual qreal getRandomReal(qreal min, qreal max) = 0;
	virtual qreal getRandomReal() = 0;

	virtual quint64 getTag() = 0;
};

#endif // RANDOMNUMBERGENERATOR_H
