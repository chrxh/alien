#ifndef UNITTHREAD_H
#define UNITTHREAD_H

#include <QThread>
#include "model/definitions.h"

class UnitThread
	: public QThread
{
	Q_OBJECT
public:
	UnitThread(QObject* parent) : QThread(parent) {}
	virtual ~UnitThread() {}
};

#endif // UNITTHREAD_H
