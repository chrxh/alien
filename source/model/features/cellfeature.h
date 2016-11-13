#ifndef CELLFEATURE_H
#define CELLFEATURE_H

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
    virtual void serialize (QDataStream& stream) const {}

protected:
    virtual ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) = 0;

    Grid* _grid;
    CellFeature* _nextFeature;

public:
    template< typename T >
    T* findObject ();
};


template< typename T >
T* CellFeature::findObject ()
{
    T* object = dynamic_cast< T* >(this);
    if( object )
        return object;
    else {
        if( _nextFeature )
            return _nextFeature->findObject<T>();
        else
            return 0;
    }
}

#endif // CELLFEATURE_H
