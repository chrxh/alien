#include "symboltable.h"

void SymbolTable::addEntry(QString const& key, QString const& value)
{
	_symbolTable[key] = value;
}

void SymbolTable::delEntry(QString const& key)
{
	_symbolTable.remove(key);
}

QString SymbolTable::applyTableToCode(QString const& input) const
{
	if (_symbolTable.contains(input)) {
		return _symbolTable[input];
	}
	return input;
}

void SymbolTable::clearTable()
{
	_symbolTable.clear();
}

QMap< QString, QString > const& SymbolTable::getTableConstRef() const
{
	return _symbolTable;
}

void SymbolTable::setTable(SymbolTable const& table)
{
	_symbolTable = table._symbolTable;
}

void SymbolTable::mergeTable(SymbolTable const& table)
{
	_symbolTable.unite(table._symbolTable);
}

void SymbolTable::serializePrimitives(QDataStream& stream) const
{
	stream << _symbolTable;
}

void SymbolTable::deserializePrimitives(QDataStream& stream)
{
	stream >> _symbolTable;
}
