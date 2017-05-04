#ifndef SIGNALWRAPPER_H
#define SIGNALWRAPPER_H

#include "model/Definitions.h"

class SignalWrapper
	: public QObject
{
	Q_OBJECT
public:
	SignalWrapper(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SignalWrapper() {}

	void emitSignal() { Q_EMIT signal(); }

signals:
	void signal();
};

#endif // SIGNALWRAPPER_H