#pragma once

#include <boost/optional.hpp>

template<typename T>
class ValueTracker
{
private:
	boost::optional<T> _value;
	boost::optional<T> _oldValue;

public:
	ValueTracker() = default;
	ValueTracker(T const& value) : _value(value), _oldValue(value) {}
	ValueTracker(boost::optional<T> const& value) : _value(value), _oldValue(value) {}
	ValueTracker(boost::optional<T> const& oldValue, boost::optional<T> const& value) : _value(value), _oldValue(oldValue) {}

	T const* operator->() const { return &*_value; }
	T* operator->() { return &*_value; }
	T const&  operator*() const { return *_value; }
	T& operator*() { return *_value; }
	explicit operator bool() const { return _value && _value != _oldValue;  }
	explicit operator boost::optional<T>() const { return _value; }

	T const& getValue() const { return *_value; }
	T & getValue() { return *_value; }
	T const& getOldValue() const { return *_oldValue; }
	T & getOldValue() { return *_oldValue; }
	ValueTracker& setValue(T const& v) { _value = v; return *this; }
};


template<typename T>
class StateTracker
{
public:
	enum class State {
		Deleted, Modified, Added
	};

private:
	State _state = State::Added;
	T _value;

public:

	StateTracker() = delete;
	StateTracker(T const &v) : _value(v) {}
	StateTracker(T const &v, State s) : _state(s), _value(v) {}

	T const* operator->() const { return &_value; }
	T* operator->() { return &_value; }

	bool isDeleted() const { return _state == State::Deleted; }
	bool isModified() const { return _state == State::Modified; }
	bool isAdded() const { return _state == State::Added; }
	StateTracker& setAsDeleted() { _state = State::Deleted; return *this; }
	StateTracker& setAsAdded() { _state = State::Added; return *this; }
	StateTracker& setAsModified() { _state = State::Modified; return *this; }
	T const& getValue() const { return _value; }
	T & getValue() { return _value; }
	StateTracker& setValue(T const& v) { _value = v; return *this; }
};

