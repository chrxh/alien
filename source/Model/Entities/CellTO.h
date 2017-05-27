#ifndef CELLTO_H
#define CELLTO_H

#include "model/Features/CellFeatureEnums.h"
#include <QVector>
#include <QList>
#include <QVector2D>
#include <QString>

class Cell;
struct MODEL_EXPORT CellTO  //TO = Transfer Object
{

    CellTO ();
    ~CellTO ();

    void copyCellProperties (const CellTO& otherCell);    //copy internal cell data except computer and token data
    void copyClusterProperties (const CellTO& otherCell); //copy internal cluster data

    //cell properties
	int numCells = 0;
    QVector2D clusterPos;
    QVector2D clusterVel;
    qreal clusterAngle = 0.0;
    qreal clusterAngVel = 0.0;
    QVector2D cellPos;
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

