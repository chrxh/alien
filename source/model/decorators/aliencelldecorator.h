#ifndef ALIENCELLDECORATOR_H
#define ALIENCELLDECORATOR_H

#include "model/entities/aliencell.h"

class AlienCellDecorator : public AlienCell
{
public:
    AlienCellDecorator (AlienCell* cell, AlienGrid*& grid) : AlienCell(grid), _cell(cell) {}
    virtual ~AlienCellDecorator ();

    template< typename T >
    static T* findObject (AlienCell* cell);

    virtual void serialize (QDataStream& stream) const;

protected:
    AlienCell* _cell;

public: //redirect following methods to AlienCell
    virtual ProcessingResult process (AlienToken* token, AlienCell* previousCell);

    virtual bool connectable (AlienCell* otherCell) const;
    virtual bool isConnectedTo (AlienCell* otherCell) const;
    virtual void resetConnections (int maxConnections);
    virtual void newConnection (AlienCell* thisCell, AlienCell* otherCell);
    virtual void delConnection (AlienCell* thisCell, AlienCell* otherCell);
    virtual void delAllConnection (AlienCell* thisCell);
    virtual int getNumConnections () const;
    virtual void setNumConnections (int num);
    virtual int getMaxConnections () const;
    virtual void setMaxConnections (int maxConnections);
    virtual AlienCell* getConnection (int i) const;
    virtual void setConnection (int i, AlienCell* cell);
    virtual QVector3D calcNormal (QVector3D outerSpace, QMatrix4x4& transform) const;

    virtual void activatingNewTokens ();
    virtual const quint64& getId () const;
    virtual void setId (quint64 id);
    virtual const quint64& getTag () const;
    virtual void setTag (quint64 tag);
    virtual int getNumToken (bool newTokenStackPointer = false) const;
    virtual AlienToken* getToken (int i) const;
    virtual void addToken (AlienToken* token, bool activateNow = true, bool setAccessNumber = true);
    virtual void delAllTokens ();

    virtual void setCluster (AlienCellCluster* cluster);
    virtual AlienCellCluster* getCluster () const;
    virtual QVector3D calcPosition (bool topologyCorrection = false) const;
    virtual void setAbsPosition (QVector3D pos);
    virtual void setAbsPositionAndUpdateMap (AlienCell* thisCell, QVector3D pos);
    virtual QVector3D getRelPos () const;
    virtual void setRelPos (QVector3D relPos);

    virtual int getTokenAccessNumber () const;
    virtual void setTokenAccessNumber (int i);
    virtual bool isTokenBlocked () const;
    virtual void setTokenBlocked (bool block);
    virtual qreal getEnergy() const;
    virtual qreal getEnergyIncludingTokens() const;
    virtual void setEnergy (qreal i);

    virtual void serialize (QDataStream& stream);

    virtual QVector3D getVel () const;
    virtual void setVel (QVector3D vel);
    virtual quint8 getColor () const;
    virtual void setColor (quint8 color);
    virtual int getProtectionCounter () const;
    virtual void setProtectionCounter (int counter);
    virtual bool isToBeKilled() const;
    virtual void setToBeKilled (bool toBeKilled);
    virtual AlienToken* takeTokenFromStack ();
};


template< typename T >
T* AlienCellDecorator::findObject (AlienCell* cell)
{
    T* object = dynamic_cast< T* >(cell);
    if( object )
        return object;
    else {
        AlienCellDecorator* decorator = dynamic_cast< AlienCellDecorator* >(cell);
        if( decorator )
            return AlienCellDecorator::findObject<T>(decorator->_cell);
        else
            return 0;
    }
}

#endif // ALIENCELLDECORATOR_H
