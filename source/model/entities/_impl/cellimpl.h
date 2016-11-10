#ifndef CELLIMPL_H
#define CELLIMPL_H

#include "model/entities/cell.h"

#include <QtGlobal>
#include <QVector3D>
#include <QVector>

class CellImpl : public Cell
{
public:

    CellImpl (qreal energy, Grid*& grid, int maxConnections = 0,
                   int tokenAccessNumber = 0, QVector3D relPos = QVector3D());
    CellImpl (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells,
                   Grid*& grid);
    ~CellImpl();

    void registerFeatureChain (CellDecorator* features);
    CellDecorator* getFeatureChain () const;

    bool connectable (Cell* otherCell) const;
    bool isConnectedTo (Cell* otherCell) const;
    void resetConnections (int maxConnections);
    void newConnection (Cell* otherCell);
    void delConnection (Cell* otherCell);
    void delAllConnection ();
    int getNumConnections () const;
    void setNumConnections (int num);
    int getMaxConnections () const;
    void setMaxConnections (int maxConnections);
    Cell* getConnection (int i) const;
    void setConnection (int i, Cell* cell);
    QVector3D calcNormal (QVector3D outerSpace, QMatrix4x4& transform) const;

    void activatingNewTokens ();
    const quint64& getId () const;
    void setId (quint64 id);
    const quint64& getTag () const;
    void setTag (quint64 tag);
    int getNumToken (bool newTokenStackPointer = false) const;
    Token* getToken (int i) const;
    void addToken (Token* token, bool activateNow = true, bool setAccessNumber = true);
    void delAllTokens ();

    void setCluster (CellCluster* cluster);
    CellCluster* getCluster () const;
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
    Token* takeTokenFromStack ();

private:
    friend class CellCluster;

    CellDecorator* _features = 0;

    QVector< Token* > _tokenStack;
    QVector< Token* > _newTokenStack;
    int _tokenStackPointer;
    int _newTokenStackPointer;

    bool _toBeKilled;
    quint64 _tag;
    quint64 _id;
    int _protectionCounter;
    QVector3D _relPos;
    CellCluster* _cluster;
    qreal _energy;

    int _maxConnections;
    int _numConnections;
    Cell** _connectingCells;

    int _tokenAccessNumber;
    bool _blockToken;

    QVector3D _vel;
    quint8 _color;      //metadata
};

#endif // CELLIMPL_H


