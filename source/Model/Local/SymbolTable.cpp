#include "SymbolTable.h"

SymbolTable::SymbolTable(QObject* parent)
	: QObject(parent)
{

}

SymbolTable::~SymbolTable()
{
}

SymbolTable * SymbolTable::clone(QObject * parent) const
{
	auto symbolTable = new SymbolTable(parent);
	symbolTable->_symbolsByKey = _symbolsByKey;
	return symbolTable;
}

void SymbolTable::addEntry(QString const& key, QString const& value)
{
	_symbolsByKey[key] = value;
}

void SymbolTable::delEntry(QString const& key)
{
	_symbolsByKey.remove(key);
}

QString SymbolTable::applyTableToCode(QString const& input) const
{
	if (_symbolsByKey.contains(input)) {
		return _symbolsByKey[input];
	}
	return input;
}

void SymbolTable::clearTable()
{
	_symbolsByKey.clear();
}

QMap< QString, QString > const& SymbolTable::getTableConstRef() const
{
	return _symbolsByKey;
}

void SymbolTable::setTable(QMap<QString, QString> const & table)
{
	_symbolsByKey = table;
}

void SymbolTable::mergeTable(SymbolTable const& table)
{
	_symbolsByKey.unite(table._symbolsByKey);
}

void SymbolTable::serializePrimitives(QDataStream& stream) const
{
	stream << _symbolsByKey;
}

void SymbolTable::deserializePrimitives(QDataStream& stream)
{
	stream >> _symbolsByKey;
}
