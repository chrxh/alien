#include "UnitThread.h"

void UnitThread::addDependency(UnitThread* unit)
{
	if (unit) {
		_dependencies.push_back(unit);
	}
}

void UnitThread::setState(State value)
{
	switch (value) {
	case State::Ready: {
		_state = value;
	} break;
	case State::Working: {
		_state = State::Working;
		for (auto const& dep : _dependencies) {
			if (dep->_state == State::Ready) {
				dep->_state = State::Working;
			}
		}
	} break;
	case State::Finished: {
		_state = value;
		for (auto const& dep : _dependencies) {
			if (dep->_state == State::Working) {
				dep->_state = State::Ready;
			}
		}
	} break;
	}
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
