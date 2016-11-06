#ifndef ALIENCELL_H
#define ALIENCELL_H

#include <QVector3D>

class AlienCellCluster;
class AlienEnergy;
class AlienGrid;
class AlienToken;

class AlienCell
{
public:
    AlienCell (AlienGrid*& grid) : _grid(grid) {}
    virtual ~AlienCell() {}

    struct ProcessingResult {
        bool decompose;
        AlienEnergy* newEnergyParticle;
    };

    virtual ProcessingResult process (AlienToken* token, AlienCell* previousCell) = 0;

    virtual bool connectable (AlienCell* otherCell) const = 0;
    virtual bool isConnectedTo (AlienCell* otherCell) const = 0;
    virtual void resetConnections (int maxConnections) = 0;
    virtual void newConnection (AlienCell* otherCell) = 0;
    virtual void delConnection (AlienCell* otherCell) = 0;
    virtual void delAllConnection () = 0;
    virtual int getNumConnections () const = 0;
    virtual void setNumConnections (int num) = 0;
    virtual int getMaxConnections () const = 0;
    virtual void setMaxConnections (int maxConnections) = 0;
    virtual AlienCell* getConnection (int i) const = 0;
    virtual void setConnection (int i, AlienCell* cell) = 0;
    virtual QVector3D calcNormal (QVector3D outerSpace, QMatrix4x4& transform) const = 0;

    virtual void activatingNewTokens () = 0;
    virtual const quint64& getId () const = 0;
    virtual void setId (quint64 id) = 0;
    virtual const quint64& getTag () const = 0;
    virtual void setTag (quint64 tag) = 0;
    virtual int getNumToken (bool newTokenStackPointer = false) const = 0;
    virtual AlienToken* getToken (int i) const = 0;
    virtual void addToken (AlienToken* token, bool activateNow = true, bool setAccessNumber = true) = 0;
    virtual void delAllTokens () = 0;

    virtual void setCluster (AlienCellCluster* cluster) = 0;
    virtual AlienCellCluster* getCluster () const = 0;
    virtual QVector3D calcPosition (bool topologyCorrection = false) const = 0;
    virtual void setAbsPosition (QVector3D pos) = 0;
    virtual void setAbsPositionAndUpdateMap (QVector3D pos) = 0;
    virtual QVector3D getRelPos () const = 0;
    virtual void setRelPos (QVector3D relPos) = 0;

    virtual int getTokenAccessNumber () const = 0;
    virtual void setTokenAccessNumber (int i) = 0;
    virtual bool isTokenBlocked () const = 0;
    virtual void setTokenBlocked (bool block) = 0;
    virtual qreal getEnergy() const = 0;
    virtual qreal getEnergyIncludingTokens() const = 0;
    virtual void setEnergy (qreal i) = 0;
    virtual QVector< quint8 >& getMemoryReference () = 0;

    virtual void serialize (QDataStream& stream) const = 0;

    virtual QVector3D getVel () const = 0;
    virtual void setVel (QVector3D vel) = 0;
    virtual quint8 getColor () const = 0;
    virtual void setColor (quint8 color) = 0;

protected:
    AlienGrid*& _grid;
};

#endif // ALIENCELL_H
