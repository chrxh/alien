#include "UnitThread.h"

void UnitThread::addDependency(UnitThread* unit)
{
	_dependencies.push_back(unit);
}

bool UnitThread::isFinished()
{
	return _state == State::Finished;
}

bool UnitThread::isReady()
{
	bool result = (_state == State::Ready);
	for (auto const& dep : _dependencies) {
		result = result && (dep->_state == State::Ready);
	}
	return result;
}
