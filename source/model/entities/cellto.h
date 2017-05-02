#ifndef CELLTO_H
#define CELLTO_H

#include "model/features/CellFeatureEnums.h"
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
	int numCells = 0;
    QVector3D clusterPos;
    QVector3D clusterVel;
    qreal clusterAngle = 0.0;
    qreal clusterAngVel = 0.0;
    QVector3D cellPos;
    qreal cellEnergy = 0.0;
    int cellNumCon = 0;
    int cellMaxCon = 0;
    bool cellAllowToken = true;
    int cellTokenAccessNum = 0;
    Enums::CellFunction::Type cellFunctionType;

    //computer data
	QByteArray computerMemory;
    QString computerCode;

    //token data
    QList< qreal > tokenEnergies;
    QList< QByteArray > tokenData;
};

#endif // CELLTO_H

