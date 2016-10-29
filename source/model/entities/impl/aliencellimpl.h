#ifndef ALIENCELL_H
#define ALIENCELL_H

#include <QtGlobal>
#include <QVector3D>

#include "alientoken.h"

class AlienCellCluster;
class AlienCellFunction;
class AlienGrid;
class AlienCell
{
public:

    static AlienCell* buildCellWithRandomData (qreal energy, AlienGrid*& grid);
    static AlienCell* buildCell (qreal energy,
                                 AlienGrid*& grid,
                                 int maxConnections = 0,
                                 int tokenAccessNumber = 0,
                                 AlienCellFunction* cellFunction = 0,
                                 QVector3D relPos = QVector3D());
    static AlienCell* buildCell (QDataStream& stream,
                                 QMap< quint64, QList< quint64 > >& connectingCells,
                                 AlienGrid*& grid);
    static AlienCell* buildCellWithoutConnectingCells (QDataStream& stream,
                                 AlienGrid*& grid);
    ~AlienCell();

    bool connectable (AlienCell* otherCell);
    bool isConnectedTo (AlienCell* otherCell);
    void resetConnections (int maxConnections);
    void newConnection (AlienCell* otherCell);
    void delConnection (AlienCell* otherCell);
    void delAllConnection ();
    int getNumConnections ();
    int getMaxConnections ();
    void setMaxConnections (int maxConnections);
    AlienCell* getConnection (int i);
    QVector3D calcNormal (QVector3D outerSpace, QMatrix4x4& transform);

    void activatingNewTokens ();
    const quint64& getId ();
    void setId (quint64 id);
    const quint64& getTag ();
    void setTag (quint64 tag);
    int getNumToken (bool newTokenStackPointer = false);
    AlienToken* getToken (int i);
    void addToken (AlienToken* token, bool activateNow = true, bool setAccessNumber = true);
    void delAllTokens ();

    void setCluster (AlienCellCluster* cluster);
    AlienCellCluster* getCluster ();
    QVector3D calcPosition (bool topologyCorrection = false);
    void setAbsPosition (QVector3D pos);
    void setAbsPositionAndUpdateMap (QVector3D pos);
    QVector3D getRelPos ();
    void setRelPos (QVector3D relPos);
    AlienCellFunction* getCellFunction ();
    void setCellFunction (AlienCellFunction* cellFunction);

    int getTokenAccessNumber ();
    void setTokenAccessNumber (int i);
    bool blockToken ();
    void setBlockToken (bool block);
    qreal getEnergy();
    qreal getEnergyIncludingTokens();
    void setEnergy (qreal i);
    QVector< quint8 >& getMemory ();

    void serialize (QDataStream& stream);

    QVector3D getVel ();
    void setVel (QVector3D vel);
    quint8 getColor ();
    void setColor (quint8 color);

private:
    friend class AlienCellCluster;

    AlienCell (qreal energy,
               AlienGrid*& grid,
               bool random = true,
               int maxConnections = 0,
               int tokenAccessNumber = 0,
               AlienCellFunction* cellFunction = 0,
               QVector3D relPos = QVector3D());
    AlienCell (QDataStream& stream,
               QMap< quint64, QList< quint64 > >& connectingCells,
               AlienGrid*& grid);
    AlienCell (QDataStream& stream, AlienGrid*& grid);     //build without connecting cells

    AlienGrid*& _grid;

    AlienCellFunction* _cellFunction;
    QVector< AlienToken* > _tokenStack;
    QVector< AlienToken* > _newTokenStack;
    int _tokenStackPointer;
    int _newTokenStackPointer;

    bool _toBeKilled;
    quint64 _tag;
    quint64 _id;
    int _protectionCounter;
    QVector3D _relPos;
    AlienCellCluster* _cluster;
    qreal _energy;

    int _maxConnections;
    int _numConnections;
    AlienCell** _connectingCells;

    int _tokenAccessNumber;
    bool _blockToken;
    QVector< quint8 > _memory;

    QVector3D _vel;
    quint8 _color;      //metadata
};

#endif // ALIENCELL_H


