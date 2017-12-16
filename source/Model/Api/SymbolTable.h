#pragma once

#include "Model/Api/Definitions.h"

class SymbolTable
	: public QObject
{
	Q_OBJECT
public:

	SymbolTable(QObject* parent = nullptr);
	virtual ~SymbolTable() = default;

	virtual SymbolTable* clone(QObject* parent = nullptr) const;

	virtual void addEntry(string const& key, string const& value);
	virtual void delEntry(string const& key);
	virtual string getValue(string const& input) const;
	virtual void clear();
	virtual map<string, string> const& getEntries () const;
	virtual void setEntries(map<string, string> const& table);
	virtual void mergeEntries(SymbolTable const& table);

private:
    map<string, string> _symbolsByKey;
};
