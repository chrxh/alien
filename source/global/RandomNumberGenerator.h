#ifndef RANDOMNUMBERGENERATOR_H
#define RANDOMNUMBERGENERATOR_H

#include "Definitions.h"

class RandomNumberGenerator
	: public QObject
{
	Q_OBJECT
public:
	RandomNumberGenerator(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~RandomNumberGenerator() = default;

	virtual void setSeed(quint32 value) = 0;

	virtual quint32 getInt() = 0;
	virtual quint32 getInt(quint32 range) = 0;
	virtual quint32 getLargeInt(quint32 range) = 0;
	virtual qreal getReal(qreal min, qreal max) = 0;
	virtual qreal getReal() = 0;
};

#endif // RANDOMNUMBERGENERATOR_H
