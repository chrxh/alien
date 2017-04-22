#ifndef NUMBERGENERATOR_H
#define NUMBERGENERATOR_H

#include <QtGlobal>
#include <QMutex>

class NumberGenerator
{
public:
	static NumberGenerator& getInstance();

	quint64 createNewTag ();
    quint64 getTag ();
    void setTag (quint64 tag);

	void setRandomSeed(quint32 value);
	quint32 random(quint32 range);
	quint32 randomLargeNumbers (quint32 range);
    qreal random (qreal min, qreal max);
	qreal random ();

private:
	NumberGenerator();
	~NumberGenerator();

	quint32 readRandomNumber();

	QMutex _mutex;
	quint64 _tag = 0;

	quint32* _arrayOfRandomNumbers = nullptr;
	int _currentNumber = 0;

};

#endif // NUMBERGENERATOR_H
