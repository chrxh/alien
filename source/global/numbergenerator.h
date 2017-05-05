#ifndef NUMBERGENERATOR_H
#define NUMBERGENERATOR_H

#include <QtGlobal>
#include <QMutex>

class NumberGenerator
{
public:
	static NumberGenerator& getInstance();

	void setSeed(quint32 value);
	quint32 getInt(quint32 range);
	quint32 getLargeInt (quint32 range);
    qreal getReal (qreal min, qreal max);
	qreal getReal ();

private:
	NumberGenerator();
	~NumberGenerator();

	quint32 readRandomNumber();

	quint32* _arrayOfRandomNumbers = nullptr;
	int _currentNumber = 0;

};

#endif // NUMBERGENERATOR_H
