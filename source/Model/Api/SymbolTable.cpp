#include "SymbolTable.h"

SymbolTable::SymbolTable(QObject* parent)
	: QObject(parent)
{

}

SymbolTable * SymbolTable::clone(QObject * parent) const
{
	auto symbolTable = new SymbolTable(parent);
	symbolTable->_symbolsByKey = _symbolsByKey;
	return symbolTable;
}

void SymbolTable::addEntry(string const& key, string const& value)
{
	_symbolsByKey[key] = value;
}

void SymbolTable::delEntry(string const& key)
{
	_symbolsByKey.erase(key);
}

string SymbolTable::getValue(string const& input) const
{
	if (_symbolsByKey.find(input) != _symbolsByKey.end()) {
		return _symbolsByKey.at(input);
	}
	return input;
}

void SymbolTable::clear()
{
	_symbolsByKey.clear();
}

map<string, string> const& SymbolTable::getEntries() const
{
	return _symbolsByKey;
}

void SymbolTable::setEntries(map<string, string> const & table)
{
	_symbolsByKey = table;
}

void SymbolTable::mergeEntries(SymbolTable const& table)
{
	_symbolsByKey.insert(table._symbolsByKey.begin(), table._symbolsByKey.end());
}

void SymbolTable::serializePrimitives(QDataStream& stream) const
{
	CHECK(false); //to be implemented
}

void SymbolTable::deserializePrimitives(QDataStream& stream)
{
	CHECK(false); //to be implemented
}
