#ifndef ALIENCELLIMPL_H
#define ALIENCELLIMPL_H

#include "model/entities/aliencell.h"

#include <QtGlobal>
#include <QVector3D>
#include <QVector>

class AlienCellImpl : public AlienCell
{
public:

    AlienCellImpl (qreal energy, AlienGrid*& grid, int maxConnections = 0,
                   int tokenAccessNumber = 0, QVector3D relPos = QVector3D());
    AlienCellImpl (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells,
                   AlienGrid*& grid);
    ~AlienCellImpl();

    ProcessingResult process (AlienToken* token, AlienCell* previousCell);

    bool connectable (AlienCell* otherCell) const;
    bool isConnectedTo (AlienCell* otherCell) const;
    void resetConnections (int maxConnections);
    void newConnection (AlienCell* otherCell);
    void delConnection (AlienCell* otherCell);
    void delAllConnection ();
    int getNumConnections () const;
    void setNumConnections (int num);
    int getMaxConnections () const;
    void setMaxConnections (int maxConnections);
    AlienCell* getConnection (int i) const;
    void setConnection (int i, AlienCell* cell);
    QVector3D calcNormal (QVector3D outerSpace, QMatrix4x4& transform) const;

    void activatingNewTokens ();
    const quint64& getId () const;
    void setId (quint64 id);
    const quint64& getTag () const;
    void setTag (quint64 tag);
    int getNumToken (bool newTokenStackPointer = false) const;
    AlienToken* getToken (int i) const;
    void addToken (AlienToken* token, bool activateNow = true, bool setAccessNumber = true);
    void delAllTokens ();

    void setCluster (AlienCellCluster* cluster);
    AlienCellCluster* getCluster () const;
    QVector3D calcPosition (bool topologyCorrection = false) const;
    void setAbsPosition (QVector3D pos);
    void setAbsPositionAndUpdateMap (QVector3D pos);
    QVector3D getRelPos () const;
    void setRelPos (QVector3D relPos);

    int getTokenAccessNumber () const;
    void setTokenAccessNumber (int i);
    bool isTokenBlocked () const;
    void setTokenBlocked (bool block);
    qreal getEnergy() const;
    qreal getEnergyIncludingTokens() const;
    void setEnergy (qreal i);

    void serialize (QDataStream& stream) const;

    QVector3D getVel () const;
    void setVel (QVector3D vel);
    quint8 getColor () const;
    void setColor (quint8 color);
    int getProtectionCounter () const;
    void setProtectionCounter (int counter);
    bool isToBeKilled() const;
    void setToBeKilled (bool toBeKilled);
    AlienToken* takeTokenFromStack ();

private:
    friend class AlienCellCluster;

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

    QVector3D _vel;
    quint8 _color;      //metadata
};

#endif // ALIENCELLIMPL_H


