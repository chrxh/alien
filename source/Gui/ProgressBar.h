#pragma once

#include <QObject>

#include "Web/Definitions.h"

#include "Definitions.h"

class QGraphicsRectItem;

class ProgressBar : public QObject
{
    Q_OBJECT

public:
    ProgressBar(std::string const& text, QWidget* parent);
    virtual ~ProgressBar();

private:
    Q_SLOT void progressStep();

    QTimer* _timer = nullptr;

    QGraphicsView* _graphicsView = nullptr;
    QGraphicsRectItem* _rectItem = nullptr;
};
