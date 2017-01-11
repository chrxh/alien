#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include "model/definitions.h"

class Symboltable
{
public:

    void addSymbolEntry (QString key, QString value);
    void delSymbolEntry (QString key);
    QString applyTableToCode (QString input) const;
    void clearTable ();
    QMap< QString, QString > const& getTable () const;
    void setTable (const QMap< QString, QString >& table);

    void serializePrimitives (QDataStream& stream) const;
    void deserializePrimitives (QDataStream& stream);

    void uniteTable(QMap< QString, QString > const& otherTable);

private:
    QMap<QString,QString> _symbolTable;
};

#endif // SYMBOLTABLE_H
