#pragma once

#include "Definitions.h"

class _PhysicalAction
{
public:
    virtual ~_PhysicalAction() = default;
};

class _ApplyForceAction : public _PhysicalAction
{
public:
    _ApplyForceAction(QVector2D const& startPos, QVector2D const& endPos, QVector2D const& force)
        : _startPos(startPos)
        , _endPos(endPos)
        , _force(force)
    {}
    virtual ~_ApplyForceAction() = default;

    QVector2D getStartPos() const { return _startPos; }
    QVector2D getEndPos() const { return _endPos; }
    QVector2D getForce() const { return _force; }

private:
    QVector2D _startPos;
    QVector2D _endPos;
    QVector2D _force;
};

class _ApplyRotationAction : public _PhysicalAction
{
public:
    _ApplyRotationAction(QVector2D const& startPos, QVector2D const& endPos, QVector2D const& force)
        : _startPos(startPos)
        , _endPos(endPos)
        , _force(force)
    {}
    virtual ~_ApplyRotationAction() = default;

    QVector2D getStartPos() const { return _startPos; }
    QVector2D getEndPos() const { return _endPos; }
    QVector2D getForce() const { return _force; }

private:
    QVector2D _startPos;
    QVector2D _endPos;
    QVector2D _force;
};


