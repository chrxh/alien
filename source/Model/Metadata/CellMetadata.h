#ifndef CELLMETADATA_H
#define CELLMETADATA_H

#include <QString>

struct CellMetadata
{
    QString computerSourcecode;
    QString name;
    QString description;
    quint8 color = 0;
};

#endif // CELLMETADATA_H
