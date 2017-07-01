#ifndef TRACKER_H
#define TRACKER_H

#include <boost/optional.hpp>

template<typename T>
class Tracker
{
private:
	boost::optional<T> _initialValue;
	boost::optional<T> _value;

public:
	Tracker() = default;
	Tracker(T const &v) : _initialValue(v), _value(v) {}

	void init(T const& v)
	{
		_initialValue = v;
		_value = v;
	}

	bool isModified() const { return _initialValue != _value; }
	bool isInitialized() const { return _value.is_initialized(); }
	T const& getValue() const { return _value.get(); }
	T const& getValueOr(T const& d) const { return _value.get_value_or(d); }
	T const& getValueOrDefault() const { return _value.get_value_or(T()); }
	T const& getInitialValue() const { return _initialValue.get(); }
	Tracker& setValue(T const& v)
	{
		_value = v;
		if (!_initialValue.is_initialized()) {
			_initialValue = v;
		}
		return *this;
	}
};

enum class TrackerElementState {
	Deleted, Retained, Added
};

template<typename T>
class TrackerElement
{
private:
	TrackerElementState _state = TrackerElementState::Retained;
	T _value;

public:
	TrackerElement() = delete;
	TrackerElement(T const &v) : _value(v) {}
	TrackerElement(T const &v, TrackerElementState s) : _state(s), _value(v) {}

	bool isDeleted() const { return _state == TrackerElementState::Deleted; }
	bool isAdded() const { return _state == TrackerElementState::Added; }
	TrackerElement& setAsDeleted() { _state = TrackerElementState::Deleted; return *this; }
	TrackerElement& setAsAdded() { _state = TrackerElementState::Added; return *this; }
	T const& getValue() const { return _value; }
	T & getValue() { return _value; }
	TrackerElement& setValue(T const& v) { _value = v; return *this; }
};



#endif // TRACKER_H
