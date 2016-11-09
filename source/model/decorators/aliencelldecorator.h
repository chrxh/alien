#ifndef ALIENCELLDECORATOR_H
#define ALIENCELLDECORATOR_H

#include <QDataStream>

class AlienGrid;
class AlienCell;
class AlienToken;
class AlienEnergy;

class AlienCellDecorator
{
public:
    AlienCellDecorator (AlienGrid*& grid) : _grid(grid), _nextFeature(0) {}
    virtual ~AlienCellDecorator ();

    void registerNextFeature (AlienCellDecorator* nextFeature);
    struct ProcessingResult {
        bool decompose;
        AlienEnergy* newEnergyParticle;
    };
    ProcessingResult process (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
    void serialize (QDataStream& stream) const;

protected:
    virtual ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell) = 0;
    virtual void serializeImpl (QDataStream& stream) const {}

    AlienGrid* _grid;
    AlienCellDecorator* _nextFeature;

public:
    template< typename T >
    static T* findObject (AlienCellDecorator* feature);
};


template< typename T >
T* AlienCellDecorator::findObject (AlienCellDecorator* feature)
{
    T* object = dynamic_cast< T* >(feature);
    if( object )
        return object;
    else {
        if( feature->_nextFeature )
            return AlienCellDecorator::findObject<T>(feature->_nextFeature);
        else
            return 0;
    }
}

#endif // ALIENCELLDECORATOR_H
