#ifndef CELLTO_H
#define CELLTO_H

#include "model/features/constants.h"
#include <QVector>
#include <QList>
#include <QVector3D>
#include <QString>

class Cell;
struct CellTO  //TO = Transfer Object
{

    CellTO ();
    ~CellTO ();

    void copyCellProperties (const CellTO& otherCell);    //copy internal cell data except computer and token data
    void copyClusterProperties (const CellTO& otherCell); //copy internal cluster data

    //cell properties
    int numCells;
    QVector3D clusterPos;
    QVector3D clusterVel;
    qreal clusterAngle;
    qreal clusterAngVel;
    QVector3D cellPos;
    qreal cellEnergy;
    int cellNumCon;
    int cellMaxCon;
    bool cellAllowToken;
    int cellTokenAccessNum;
    CellFunctionType cellFunctionType;

    //computer data
    QVector< quint8 > computerMemory;
    QString computerCode;

    //token data
    QList< qreal > tokenEnergies;
    QList< QVector< quint8 > > tokenData;
};

#endif // CELLTO_H

