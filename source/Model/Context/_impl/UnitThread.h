#ifndef UNITTHREAD_H
#define UNITTHREAD_H

#include <QThread>
#include "Model/Definitions.h"

class MODEL_EXPORT UnitThread
	: public QThread
{
	Q_OBJECT
public:
	UnitThread(QObject* parent = nullptr) : QThread(parent) {}
	virtual ~UnitThread() {}

	void addDependency(UnitThread* unit);

	enum class State { Ready, Working, Finished };
	void setState(State value);
	bool isReady();
	bool isWorking();
	bool isFinished();

private:
	State _state = State::Ready;
	vector<UnitThread*> _dependencies;
};

#endif // UNITTHREAD_H
