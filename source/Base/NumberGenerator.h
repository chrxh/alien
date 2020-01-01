#pragma once

#include "Definitions.h"

class NumberGenerator
	: public QObject
{
	Q_OBJECT
public:
	NumberGenerator(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~NumberGenerator() = default;

	virtual void init(uint32_t arraySize = 1323781, uint16_t threadId = 0) = 0;

	virtual uint32_t getRandomInt() = 0;
	virtual uint32_t getRandomInt(uint32_t range) = 0;
	virtual uint32_t getLargeRandomInt(uint32_t range) = 0;
	virtual double getRandomReal(double min, double max) = 0;
	virtual double getRandomReal() = 0;
	virtual QByteArray getRandomArray(int length) = 0;

	virtual uint64_t getId() = 0;
};
