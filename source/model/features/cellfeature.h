#ifndef CELLDECORATOR_H
#define CELLDECORATOR_H

#include <QDataStream>

class Grid;
class Cell;
class Token;
class EnergyParticle;

class CellFeature
{
public:
    CellFeature (Grid*& grid) : _grid(grid), _nextFeature(0) {}
    virtual ~CellFeature ();

    void registerNextFeature (CellFeature* nextFeature);
    struct ProcessingResult {
        bool decompose;
        EnergyParticle* newEnergyParticle;
    };
    virtual ProcessingResult process (Token* token, Cell* cell, Cell* previousCell);
    virtual void serialize (QDataStream& stream) const;

protected:
    virtual ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) = 0;
    virtual void serializeImpl (QDataStream& stream) const {}

    Grid* _grid;
    CellFeature* _nextFeature;

public:
    template< typename T >
    static T* findObject (CellFeature* feature);
};


template< typename T >
T* CellFeature::findObject (CellFeature* feature)
{
    T* object = dynamic_cast< T* >(feature);
    if( object )
        return object;
    else {
        if( feature->_nextFeature )
            return CellFeature::findObject<T>(feature->_nextFeature);
        else
            return 0;
    }
}

#endif // CELLDECORATOR_H
