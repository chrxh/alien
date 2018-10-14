#pragma once

#include <QThread>

#include "Definitions.h"

class MODELCPU_EXPORT UnitThread
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
