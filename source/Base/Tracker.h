#ifndef TRACKER_H
#define TRACKER_H

#include <boost/optional.hpp>
#include <vector>

using std::vector;

template<typename T>
class ChangeTracker
{
public:
	enum class State {
		Deleted, Modified, Added
	};

private:
	State _state = State::Added;
	T _value;

public:

	ChangeTracker() = delete;
	ChangeTracker(T const &v) : _value(v) {}
	ChangeTracker(T const &v, State s) : _state(s), _value(v) {}

	T const* operator->() const { return &_value; }
	T* operator->() { return &_value; }

	bool isDeleted() const { return _state == State::Deleted; }
	bool isModified() const { return _state == State::Modified; }
	bool isAdded() const { return _state == State::Added; }
	ChangeTracker& setAsDeleted() { _state = State::Deleted; return *this; }
	ChangeTracker& setAsAdded() { _state = State::Added; return *this; }
	ChangeTracker& setAsModified() { _state = State::Modified; return *this; }
	T const& getValue() const { return _value; }
	T & getValue() { return _value; }
	ChangeTracker& setValue(T const& v) { _value = v; return *this; }
};

#endif // TRACKER_H
