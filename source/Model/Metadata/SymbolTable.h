#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include "model/Definitions.h"

class SymbolTable
	: public QObject
{
	Q_OBJECT
public:

	SymbolTable(QObject* parent = nullptr);
	virtual ~SymbolTable();

	virtual SymbolTable* clone(QObject* parent = nullptr) const;

	virtual void addEntry(QString const& key, QString const& value);
	virtual void delEntry(QString const& key);
	virtual QString applyTableToCode(QString const& input) const;
	virtual void clearTable();
	virtual QMap< QString, QString > const& getTableConstRef () const;
	virtual void setTable(SymbolTable const& table);
	virtual void mergeTable(SymbolTable const& table);

	virtual void serializePrimitives (QDataStream& stream) const;
	virtual void deserializePrimitives (QDataStream& stream);

private:
    QMap<QString,QString> _symbolsByKey;
};

#endif // SYMBOLTABLE_H
