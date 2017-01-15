#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include "model/definitions.h"

class SymbolTable
{
public:

    void addEntry(QString const& key, QString const& value);
    void delEntry(QString const& key);
    QString applyTableToCode(QString const& input) const;
    void clearTable();
    QMap< QString, QString > const& getTableConstRef () const;
    void setTable(SymbolTable const& table);
	void mergeTable(SymbolTable const& table);

    void serializePrimitives (QDataStream& stream) const;
    void deserializePrimitives (QDataStream& stream);

private:
    QMap<QString,QString> _symbolTable;
};

#endif // SYMBOLTABLE_H
