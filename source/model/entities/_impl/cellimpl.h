#ifndef CELLIMPL_H
#define CELLIMPL_H

#include "model/entities/cell.h"

#include <QtGlobal>
#include <QVector3D>
#include <QVector>

class CellImpl : public Cell
{
public:

    CellImpl (Grid* grid);
    CellImpl (qreal energy, Grid* grid, int maxConnections = 0,
                   int tokenAccessNumber = 0, QVector3D relPos = QVector3D());
    CellImpl (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells,
                   Grid* grid);
    ~CellImpl();

    bool compareEqual (Cell* otherCell) const;

    void registerFeatures (CellFeature* features);
    CellFeature* getFeatures () const;
    void removeFeatures ();

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
    QVector3D calcNormal (QVector3D outerSpace) const;

    void activatingNewTokens ();
    const quint64& getId () const;
    void setId (quint64 id);
    const quint64& getTag () const;
    void setTag (quint64 tag);
    int getNumToken (bool newTokenStackPointer = false) const;
    Token* getToken (int i) const;
    void addToken (Token* token, ACTIVATE_TOKEN act, UPDATE_TOKEN_ACCESS_NUMBER update);
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

    QVector3D getVel () const;
    void setVel (QVector3D vel);
    quint8 getColor () const;
    void setColor (quint8 color);
    int getProtectionCounter () const;
    void setProtectionCounter (int counter);
    bool isToBeKilled() const;
    void setToBeKilled (bool toBeKilled);
    int getTokenStackPointer () const override;
    QVector<Token*>& getTokenStackRef () override;
    Token* takeTokenFromStack () override;

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives(QDataStream& stream) override;


private:
    friend class CellCluster;

    CellFeature* _features = nullptr;

    QVector< Token* > _tokenStack;
    QVector< Token* > _newTokenStack;
    int _tokenStackPointer = 0;
    int _newTokenStackPointer = 0;

    bool _toBeKilled = false;
    quint64 _tag = 0;
    quint64 _id = 0;
    int _protectionCounter = 0;
    QVector3D _relPos;
    CellCluster* _cluster = nullptr;
    qreal _energy = 0.0;

    int _maxConnections = 0;
    int _numConnections = 0;
    Cell** _connectingCells = nullptr;

    int _tokenAccessNumber = 0;
    bool _blockToken = false;

    QVector3D _vel;
    quint8 _color = 0;      //metadata
};

#endif // CELLIMPL_H


