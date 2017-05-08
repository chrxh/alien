#ifndef EDITABLE_H
#define EDITABLE_H

template<typename T>
class Editable
{
private:
	bool _initialized = false;
	T _initialValue;
	T _value;

public:
	Editable() = default;
	Editable(T const &v) : _initialized(true), _initialValue(v), _value(v) {}

	void init(T const& v)
	{
		_initialValue = v;
		_value = v;
		_initialized = true;
	}

	void check() const
	{
		if (!isInitialized()) {
			throw std::exception("Editable is not initialized.");
		}
	}

	bool isInitialized() const { return _initialized; }
	bool isModified() const { return _initialValue != _value; }
	T const& getValue() const { check(); return _value; }
	T const& getValueOrDefault(T const& d) const
	{
		if (isInitialized()) {
			return _value;
		}
		return d;
	}
	T const& getInitialValue() const { check(); return _initialValue; }
	Editable& setValue(T const& v) { check(); _value = v; return *this; }
};

enum class EditableVecState {
	Deleted, Maintained, Added
};

template<typename T>
class EditableVec
{
private:
	EditableVecState _state = EditableVecState::Maintained;
	T _value;

public:
	EditableVec() = default;
	EditableVec(T const &v) : _value(v) {}
	EditableVec(EditableVecState s, T const &v) : _state(s), _value(v) {}

private:
	void check() const
	{
		if (isDeleted()) {
			throw std::exception("The value of a deleted object cannot be accessed.");
		}
	}

public:
	bool isDeleted() const { return _state == EditableVecState::Deleted; }
	bool isAdded() const { return _state == EditableVecState::Added; }
	EditableVec& markAsDeleted() { _state = EditableVecState::Deleted; return *this; }
	EditableVec& markAsAdded() { _state = EditableVecState::Added; return *this; }
	T const& getValue() const { check(); return _value; }
	EditableVec& setValue(T const& v) { check(); _value = v; return *this; }
};



#endif // EDITABLE_H
