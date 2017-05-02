#ifndef UNITTHREAD_H
#define UNITTHREAD_H

#include <QThread>
#include "model/Definitions.h"

class UnitThread
	: public QThread
{
	Q_OBJECT
public:
	UnitThread(QObject* parent) : QThread(parent) {}
	virtual ~UnitThread() {}

	void addDependency(UnitThread* unit);

	enum class State { Ready, Working, Finished };
	void setState(State value);
	bool isFinished();
	bool isReady();

private:
	State _state = State::Ready;
	std::vector<UnitThread*> _dependencies;
};

#endif // UNITTHREAD_H
