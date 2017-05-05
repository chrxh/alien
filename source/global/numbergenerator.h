#ifndef NUMBERGENERATOR_H
#define NUMBERGENERATOR_H

#include <QtGlobal>
#include <QMutex>

class NumberGenerator
{
public:
	static NumberGenerator& getInstance();

	void setRandomSeed(quint32 value);
	quint32 random(quint32 range);
	quint32 randomLargeNumbers (quint32 range);
    qreal random (qreal min, qreal max);
	qreal random ();

private:
	NumberGenerator();
	~NumberGenerator();

	quint32 readRandomNumber();

	quint32* _arrayOfRandomNumbers = nullptr;
	int _currentNumber = 0;

};

#endif // NUMBERGENERATOR_H
